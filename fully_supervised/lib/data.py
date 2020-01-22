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

import os

from absl import flags

from libml import augment as augment_module
from libml import data

FLAGS = flags.FLAGS


class DataSetsFS(data.DataSets):
    @classmethod
    def creator(cls, name, train_files, test_files, valid, augment, parse_fn=data.record_parse, do_memoize=True,
                nclass=10, height=32, width=32, colors=3):
        train_files = [os.path.join(data.DATA_DIR, x) for x in train_files]
        test_files = [os.path.join(data.DATA_DIR, x) for x in test_files]
        if not isinstance(augment, list):
            augment = augment(name)
        else:
            assert len(augment) == 1
            augment = augment[0]

        def create():
            image_shape = [height, width, colors]
            kwargs = dict(parse_fn=parse_fn, image_shape=image_shape)
            train_labeled = data.DataSet.from_files(train_files, augment, **kwargs)
            if do_memoize:
                train_labeled = train_labeled.memoize()
            if FLAGS.whiten:
                mean, std = data.compute_mean_std(train_labeled)
            else:
                mean, std = 0, 1

            valid_data = data.DataSet.from_files(train_files, augment_module.NOAUGMENT, **kwargs).take(valid)
            test_data = data.DataSet.from_files(test_files, augment_module.NOAUGMENT, **kwargs)

            return cls(name + '.' + FLAGS.augment.split('.')[0] + '-' + str(valid),
                       train_labeled=train_labeled.skip(valid),
                       train_unlabeled=None,
                       valid=valid_data,
                       test=test_data,
                       nclass=nclass, colors=colors, height=height, width=width, mean=mean, std=std)

        return name + '-' + str(valid), create


def augment_function(dataset: str):
    return augment_module.get_augmentation(dataset, FLAGS.augment.split('.')[0])


def create_datasets():
    d = {}
    d.update([DataSetsFS.creator('cifar10', ['cifar10-train.tfrecord'], ['cifar10-test.tfrecord'], valid,
                                 augment_function) for valid in [1, 5000]])
    d.update([DataSetsFS.creator('cifar100', ['cifar100-train.tfrecord'], ['cifar100-test.tfrecord'], valid,
                                 augment_function, nclass=100) for valid in [1, 5000]])
    d.update([DataSetsFS.creator('fashion_mnist', ['fashion_mnist-train.tfrecord'], ['fashion_mnist-test.tfrecord'],
                                 valid, augment_function, height=32, width=32, colors=1,
                                 parse_fn=data.record_parse_mnist)
              for valid in [1, 5000]])
    d.update(
        [DataSetsFS.creator('stl10', [], [], valid, augment_function, height=96, width=96, do_memoize=False)
         for valid in [1, 5000]])
    d.update([DataSetsFS.creator('svhn', ['svhn-train.tfrecord', 'svhn-extra.tfrecord'], ['svhn-test.tfrecord'],
                                 valid, augment_function, do_memoize=False) for valid in [1, 5000]])
    d.update([DataSetsFS.creator('svhn_noextra', ['svhn-train.tfrecord'], ['svhn-test.tfrecord'],
                                 valid, augment_function, do_memoize=False) for valid in [1, 5000]])
    return d


DATASETS = create_datasets
