# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input data for image models.
"""

import functools
import itertools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import tqdm

from libml import augment as augment_module
from libml import utils
from libml.augment import AugmentPair, NOAUGMENT

# Data directory. Value is initialized in _data_setup
#
# Note that if you need to use DATA_DIR outside of this module then
# you should do following:
#     from libml import data as libml_data
#     ...
#     dir = libml_data.DATA_DIR
#
# If you directly import DATA_DIR:
#   from libml.data import DATA_DIR
# then None will be imported.
DATA_DIR = None

_DATA_CACHE = None
SAMPLES_PER_CLASS = [1, 2, 3, 4, 5, 10, 25, 100, 400]

flags.DEFINE_string('dataset', 'cifar10.1@4000-5000', 'Data to train on.')
flags.DEFINE_integer('para_parse', 1, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 5, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_string('p_unlabeled', '', 'Probability distribution of unlabeled.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
flags.DEFINE_string('data_dir', None,
                    'Data directory. '
                    'If None then environment variable ML_DATA '
                    'will be used as a data directory.')

FLAGS = flags.FLAGS


def _data_setup():
    # set up data directory
    global DATA_DIR
    DATA_DIR = FLAGS.data_dir or os.environ['ML_DATA']


app.call_after_init(_data_setup)


def record_parse_mnist(serialized_example, image_shape=None):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    if image_shape:
        image.set_shape(image_shape)
    image = tf.pad(image, [[2] * 2, [2] * 2, [0] * 2])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(image=image, label=features['label'])


def record_parse(serialized_example, image_shape=None):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    if image_shape:
        image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(image=image, label=features['label'])


def compute_mean_std(data: tf.data.Dataset):
    data = data.map(lambda x: x['image']).batch(1024).prefetch(1)
    data = data.make_one_shot_iterator().get_next()
    count = 0
    stats = []
    with tf.Session(config=utils.get_config()) as sess:
        def iterator():
            while True:
                try:
                    yield sess.run(data)
                except tf.errors.OutOfRangeError:
                    break

        for batch in tqdm(iterator(), unit='kimg', desc='Computing dataset mean and std'):
            ratio = batch.shape[0] / 1024.
            count += ratio
            stats.append((batch.mean((0, 1, 2)) * ratio, (batch ** 2).mean((0, 1, 2)) * ratio))
    mean = sum(x[0] for x in stats) / count
    sigma = sum(x[1] for x in stats) / count - mean ** 2
    std = np.sqrt(sigma)
    print('Mean %s  Std: %s' % (mean, std))
    return mean, std


class DataSet:
    """Wrapper for tf.data.Dataset to permit extensions."""

    def __init__(self, data: tf.data.Dataset, augment_fn: AugmentPair, parse_fn=record_parse, image_shape=None):
        self.data = data
        self.parse_fn = parse_fn
        self.augment_fn = augment_fn
        self.image_shape = image_shape

    @classmethod
    def from_files(cls, filenames: list, augment_fn: AugmentPair, parse_fn=record_parse, image_shape=None):
        filenames_in = filenames
        filenames = sorted(sum([tf.gfile.Glob(x) for x in filenames], []))
        if not filenames:
            raise ValueError('Empty dataset, did you mount gcsfuse bucket?', filenames_in)
        if len(filenames) > 4:
            def fetch_dataset(filename):
                buffer_size = 8 * 1024 * 1024  # 8 MiB per file
                dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
                return dataset

            # Read the data from disk in parallel
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.apply(
                tf.data.experimental.parallel_interleave(
                    fetch_dataset,
                    cycle_length=min(16, len(filenames)),
                    sloppy=True))
        else:
            dataset = tf.data.TFRecordDataset(filenames)
        return cls(dataset,
                   augment_fn=augment_fn,
                   parse_fn=parse_fn,
                   image_shape=image_shape)

    @classmethod
    def empty_data(cls, image_shape, augment_fn: AugmentPair = None):
        def _get_null_input(_):
            return dict(image=tf.zeros(image_shape, tf.float32),
                        label=tf.constant(0, tf.int64))

        return cls(tf.data.Dataset.range(FLAGS.batch).map(_get_null_input),
                   parse_fn=None,
                   augment_fn=augment_fn,
                   image_shape=image_shape)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        def call_and_update(*args, **kwargs):
            v = getattr(self.__dict__['data'], item)(*args, **kwargs)
            if isinstance(v, tf.data.Dataset):
                return self.__class__(v,
                                      parse_fn=self.parse_fn,
                                      augment_fn=self.augment_fn,
                                      image_shape=self.image_shape)
            return v

        return call_and_update

    def parse(self):
        if self.parse_fn:
            para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
            if self.image_shape:
                return self.map(lambda x: self.parse_fn(x, self.image_shape), para)
            else:
                return self.map(self.parse_fn, para)
        return self

    def numpy_augment(self, *args, **kwargs):
        return self.augment_fn.numpy(*args, **kwargs)

    def augment(self):
        if self.augment_fn:
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            return self.map(self.augment_fn.tf, para)
        return self

    def memoize(self):
        """Call before parsing, since it calls for parse inside."""
        data = []
        with tf.Session(config=utils.get_config()) as session:
            it = self.parse().prefetch(16).make_one_shot_iterator().get_next()
            try:
                while 1:
                    data.append(session.run(it))
            except tf.errors.OutOfRangeError:
                pass
        images = np.stack([x['image'] for x in data])
        labels = np.stack([x['label'] for x in data])

        def tf_get(index, image_shape):
            def get(index):
                return images[index], labels[index]

            image, label = tf.py_func(get, [index], [tf.float32, tf.int64])
            return dict(image=tf.reshape(image, image_shape), label=label, index=index)

        return self.__class__(tf.data.Dataset.range(len(data)),
                              parse_fn=tf_get,
                              augment_fn=self.augment_fn,
                              image_shape=self.image_shape)


class DataSets:
    def __init__(self, name, train_labeled: DataSet, train_unlabeled: DataSet, test: DataSet, valid: DataSet,
                 height=32, width=32, colors=3, nclass=10, mean=0, std=1, p_labeled=None, p_unlabeled=None):
        self.name = name
        self.train_labeled = train_labeled
        self.train_unlabeled = train_unlabeled
        self.test = test
        self.valid = valid
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.p_labeled = p_labeled
        self.p_unlabeled = p_unlabeled

    @classmethod
    def creator(cls, name, seed, label, valid, augment, parse_fn=record_parse, do_memoize=False,
                nclass=10, colors=3, height=32, width=32):
        if not isinstance(augment, list):
            augment = augment(name)
        fullname = '.%d@%d' % (seed, label)
        root = os.path.join(DATA_DIR, 'SSL2', name)

        def create():
            p_labeled = p_unlabeled = None

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            image_shape = [height, width, colors]
            train_labeled = DataSet.from_files(
                [root + fullname + '-label.tfrecord'], augment[0], parse_fn, image_shape)
            train_unlabeled = DataSet.from_files(
                [root + '-unlabel.tfrecord'], augment[1], parse_fn, image_shape)
            if do_memoize:
                train_labeled = train_labeled.memoize()
                train_unlabeled = train_unlabeled.memoize()

            if FLAGS.whiten:
                mean, std = compute_mean_std(train_labeled.concatenate(train_unlabeled))
            else:
                mean, std = 0, 1

            test_data = DataSet.from_files(
                [os.path.join(DATA_DIR, '%s-test.tfrecord' % name)], NOAUGMENT, parse_fn, image_shape=image_shape)

            return cls(name + '.' + FLAGS.augment + fullname + '-' + str(valid)
                       + ('/' + FLAGS.p_unlabeled if FLAGS.p_unlabeled else ''),
                       train_labeled=train_labeled,
                       train_unlabeled=train_unlabeled.skip(valid),
                       valid=train_unlabeled.take(valid),
                       test=test_data,
                       nclass=nclass, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       height=height, width=width, colors=colors, mean=mean, std=std)

        return name + fullname + '-' + str(valid), create


def create_datasets(augment_fn):
    d = {}
    d.update([DataSets.creator('cifar10', seed, label, valid, augment_fn)
              for seed, label, valid in itertools.product(range(6), [10 * x for x in SAMPLES_PER_CLASS], [1, 5000])])
    d.update([DataSets.creator('cifar100', seed, label, valid, augment_fn, nclass=100)
              for seed, label, valid in itertools.product(range(6), [400, 1000, 2500, 10000], [1, 5000])])
    d.update([DataSets.creator('fashion_mnist', seed, label, valid, augment_fn, height=32, width=32, colors=1,
                               parse_fn=record_parse_mnist)
              for seed, label, valid in itertools.product(range(6), [10 * x for x in SAMPLES_PER_CLASS], [1, 5000])])
    d.update([DataSets.creator('stl10', seed, label, valid, augment_fn, height=96, width=96)
              for seed, label, valid in itertools.product(range(6), [1000, 5000], [1, 500])])
    d.update([DataSets.creator('svhn', seed, label, valid, augment_fn)
              for seed, label, valid in itertools.product(range(6), [10 * x for x in SAMPLES_PER_CLASS], [1, 5000])])
    d.update([DataSets.creator('svhn_noextra', seed, label, valid, augment_fn)
              for seed, label, valid in itertools.product(range(6), [10 * x for x in SAMPLES_PER_CLASS], [1, 5000])])
    return d


DATASETS = functools.partial(create_datasets, augment_module.augment_function)
PAIR_DATASETS = functools.partial(create_datasets, augment_module.pair_augment_function)
MANY_DATASETS = functools.partial(create_datasets, augment_module.many_augment_function)
QUAD_DATASETS = functools.partial(create_datasets, augment_module.quad_augment_function)
