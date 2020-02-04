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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

import os
from absl import flags

import tensorflow as tf


IMAGENET_BUFFER_SIZE = 16 * 1024 * 1024     # 16 MiB
IMAGENET_FETCH_CYCLE_LENGTH = 16
IMAGENET_SHUFFLE_BUFFER_SIZE = 1024
IMAGENET_PREPROCESSING_THREADS = 2

IMAGE_SIZE = 224
CROP_PADDING = 32

NUM_TRAINING_IMAGES = 1281167
NUM_TEST_IMAGES = 50000
NUM_CLASSES = 1000


SPLIT_TO_FILE_PATTERN = {
    'train': 'train-*',
    'test': 'validation-*',
    'train128116.1': 'train128116.1-*',
    'train128116.2': 'train128116.2-*',
    'train128116.3': 'train128116.3-*',
    'train128116.4': 'train128116.4-*',
    'train128116.5': 'train128116.5-*',
    'train64058.1': 'train64058.1-*',
    'train64058.2': 'train64058.2-*',
    'train64058.3': 'train64058.3-*',
    'train64058.4': 'train64058.4-*',
    'train64058.5': 'train64058.5-*',
    'train12811.1': 'train12811.1-*',
    'train12811.2': 'train12811.2-*',
    'train12811.3': 'train12811.3-*',
    'train12811.4': 'train12811.4-*',
    'train12811.5': 'train12811.5-*',
}


SPLIT_TO_NUM_IMAGES = {
    'train': 1281167,
    'test': 50000,
    'train128116.1': 128116,
    'train128116.2': 128116,
    'train128116.3': 128116,
    'train128116.4': 128116,
    'train128116.5': 128116,
    'train64058.1': 64058,
    'train64058.2': 64058,
    'train64058.3': 64058,
    'train64058.4': 64058,
    'train64058.5': 64058,
    'train12811.1': 12811,
    'train12811.2': 12811,
    'train12811.3': 12811,
    'train12811.4': 12811,
    'train12811.5': 12811,
}


flags.DEFINE_string('imagenet_data', None, 'Path to imagenet data.')

FLAGS = flags.FLAGS


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope('distorted_bounding_box_crop'):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes):
  """Make a random crop of IMAGE_SIZE."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes),
      lambda: tf.image.resize([image],  # pylint: disable=g-long-lambda
                              [IMAGE_SIZE, IMAGE_SIZE],
                              method=tf.image.ResizeMethod.BICUBIC)[0])

  return image


def _decode_and_center_crop(image_bytes):
  """Crops to center of image with padding then scales IMAGE_SIZE."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize([image],
                          [IMAGE_SIZE, IMAGE_SIZE],
                          method=tf.image.ResizeMethod.BICUBIC)[0]

  return image


def preprocess_image(image_bytes,
                     is_training=False,
                     augmentation=None,
                     use_bfloat16=False,
                     saturate_uint8=False,
                     scale_and_center=False,
                     use_default_augment=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    augmentation: callable which performs augmentation on images.
    use_bfloat16: `bool` for whether to use bfloat16.
    saturate_uint8: If True then perform saturate cast to uint8
      before augmentation.
    scale_and_center: If True then rescale image to [-1, 1] range
      after augmentation.
    use_default_augment: If True then apply defaul augment (left-right flip)
        before main augmentation on all training images.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    image = _decode_and_random_crop(image_bytes)
  else:
    image = _decode_and_center_crop(image_bytes)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  # decode and crop returns float32 image with values in range [0, 255]
  if saturate_uint8:
    image = tf.saturate_cast(image, tf.uint8)
  # do augmentations if necessary
  if use_default_augment and is_training:
    image = tf.image.random_flip_left_right(image)
  if augmentation is not None:
    tensors_dict = augmentation(image)
  else:
    tensors_dict = {'image': image}
  # cast and rescale all image tensors
  dtype = tf.bfloat16 if use_bfloat16 else tf.float32
  for k, v in tensors_dict.items():
    if k.endswith('image'):
      v = tf.cast(v, dtype)
      if scale_and_center:
        v = v / tf.constant(127.5, dtype) - tf.constant(1.0, dtype)
      tensors_dict[k] = v
  return tensors_dict


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               split,
               is_training,
               batch_size,
               augmentation=None,
               use_bfloat16=False,
               saturate_uint8=False,
               scale_and_center=False,
               use_default_augment=False):
    """Initialize ImageNetInput.

    Args:
      split: data split, either 'train' or 'test'.
      is_training: `bool` for whether the input is for training.
      batch_size: The global batch size to use.
      augmentation: callable which performs augmentation on images.
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      saturate_uint8: If True then perform saturate cast to uint8
        before augmentation.
      scale_and_center: If True then rescale image to [-1, 1] range
        after augmentation.
      use_default_augment: If True then apply defaul augment (left-right flip)
        before main augmentation on all training images.
    """
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.saturate_uint8 = saturate_uint8
    self.scale_and_center = scale_and_center
    self.use_default_augment = use_default_augment
    self.batch_size = batch_size
    self.augmentation = augmentation
    self.num_classes = NUM_CLASSES
    self.num_images = SPLIT_TO_NUM_IMAGES[split.lower()]
    self.file_pattern = os.path.join(FLAGS.imagenet_data,
                                     SPLIT_TO_FILE_PATTERN[split.lower()])

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.io.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    tensors_dict = preprocess_image(
        image_bytes=image_bytes,
        is_training=self.is_training,
        augmentation=self.augmentation,
        use_bfloat16=self.use_bfloat16,
        saturate_uint8=self.saturate_uint8,
        scale_and_center=self.scale_and_center,
        use_default_augment=self.use_default_augment)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(tf.reshape(parsed['image/class/label'], shape=()) - 1,
                    dtype=tf.int32)
    tensors_dict['label'] = label

    return tensors_dict

  def make_parsed_dataset(self, ctx=None):
    """Helper function which makes tf.Dataset object of parsed records."""
    # Shuffle the filenames to ensure better randomization.
    dataset = tf.data.Dataset.list_files(self.file_pattern,
                                         shuffle=self.is_training)

    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      return tf.data.TFRecordDataset(filename, buffer_size=IMAGENET_BUFFER_SIZE)

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        fetch_dataset, cycle_length=IMAGENET_FETCH_CYCLE_LENGTH,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self.is_training:
      dataset = dataset.shuffle(IMAGENET_SHUFFLE_BUFFER_SIZE)

    # Parse, pre-process, and batch the data in parallel
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=self.batch_size,
            num_parallel_batches=IMAGENET_PREPROCESSING_THREADS,
            drop_remainder=self.is_training))

    return dataset

  def input_fn(self, ctx=None):
    """Input function which provides a single batch for train or eval.

    Args:
      ctx: Input context.

    Returns:
      A `tf.data.Dataset` object.
    """
    dataset = self.make_parsed_dataset(ctx)

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self.is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      dataset = dataset.with_options(options)

    return dataset


def _combine_sup_unsup_datasets(sup_data, unsup_data):
  """Combines supervised and usupervised samples into single dictionary.

  Args:
    sup_data: dictionary with examples from supervised dataset.
    unsup_data: dictionary with examples from unsupervised dataset.

  Returns:
    Dictionary with combined suvervised and unsupervised examples.
  """
  # Copy all values from supervised data as is
  output_dict = dict(sup_data)

  # take only 'image' and 'aug_image' from unsupervised dataset and
  # rename then into 'unsup_image' and 'unsup_aug_image'
  if 'image' in unsup_data:
    output_dict['unsup_image'] = unsup_data.pop('image')
  if 'aug_image' in unsup_data:
    output_dict['unsup_aug_image'] = unsup_data.pop('aug_image')

  return output_dict


class ImageNetSslTrainInput(object):
  """Generates Imagenet input_fn for semi-supervised training."""

  def __init__(self,
               supervised_split,
               supervised_batch_size,
               unsupervised_batch_size,
               supervised_augmentation,
               unsupervised_augmentation,
               **kwargs):
    """Initialize ImageNetSslTrainInput.

    Args:
      supervised_split: split of supervised data.
      supervised_batch_size: batch size for supervised data.
      unsupervised_batch_size: batch size for unsupervised data.
      supervised_augmentation: augmentation for supervised data.
      unsupervised_augmentation: augmentation for unsupervised data.
      **kwargs: other arguments which are directly passed to ImageNetInput.
    """
    self.supervised_input = ImageNetInput(
        split=supervised_split,
        is_training=True,
        batch_size=supervised_batch_size,
        augmentation=supervised_augmentation,
        **kwargs)
    self.unsupervised_input = ImageNetInput(
        split='train',
        is_training=True,
        batch_size=unsupervised_batch_size,
        augmentation=unsupervised_augmentation,
        **kwargs)
    self.batch_size = self.supervised_input.batch_size
    self.num_images = self.supervised_input.num_images
    self.num_classes = NUM_CLASSES

  def input_fn(self, ctx=None):
    """Input function which provides a single batch for training.

    Args:
      ctx: Input context.

    Returns:
      A `tf.data.Dataset` object.
    """
    sup_dataset = self.supervised_input.make_parsed_dataset(ctx)
    unsup_dataset = self.unsupervised_input.make_parsed_dataset(ctx)

    dataset = tf.data.Dataset.zip((sup_dataset, unsup_dataset))
    dataset = dataset.map(_combine_sup_unsup_datasets)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

    return dataset
