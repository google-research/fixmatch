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
"""Code to create datasets."""

import collections
import math

from absl import logging

from datasets import imagenet


Datasets = collections.namedtuple('Datasets',
                                  ['train_dataset',
                                   'eval_dataset',
                                   'num_classes',
                                   'steps_per_epoch',
                                   'steps_per_eval'])


def _num_workers(distribution_strategy):
  """Returns number of workers in TPU pod or 1 for TPU donut."""
  # Input pipeline logit is slightly different for TPU pod and donut.
  # This function is a helper to determine number of devices to distribute
  # input pipeline on.
  # For TPU donut (2x2 topology) input pipeline is automatically distributed
  # by distribution strategy, thus thus function returns one.
  # For larger slices of TPU pod, input pipeline is manually distributed across
  # number of devices returned by this function.
  is_tpu_pod = distribution_strategy.extended._input_workers.num_workers > 1  # pylint: disable=protected-access
  if is_tpu_pod:
    return len(distribution_strategy.extended.worker_devices)
  else:
    return 1


def _make_datasets_namedtuple(strategy,
                              train_dataset_object,
                              eval_dataset_object):
  """Makes Datasets namedtuple from objects which provide input_fn.

  Args:
    strategy: distribution strategy.
    train_dataset_object: object which provides input_fn for training data.
    eval_dataset_object: object which provides input_fn for evaluation data.

  Returns:
    Datasets namedtuple with train and eval datasets.
  """
  num_workers = _num_workers(strategy)
  if num_workers > 1:
    train_dataset = strategy.experimental_distribute_datasets_from_function(
        train_dataset_object.input_fn)
    eval_dataset = strategy.experimental_distribute_datasets_from_function(
        eval_dataset_object.input_fn)
  else:
    train_dataset = strategy.experimental_distribute_dataset(
        train_dataset_object.input_fn())
    eval_dataset = strategy.experimental_distribute_dataset(
        eval_dataset_object.input_fn())

  train_batch_size = train_dataset_object.batch_size * num_workers
  steps_per_epoch = train_dataset_object.num_images / train_batch_size

  eval_batch_size = eval_dataset_object.batch_size * num_workers
  steps_per_eval = int(
      math.ceil(eval_dataset_object.num_images / eval_batch_size))

  return Datasets(train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  num_classes=train_dataset_object.num_classes,
                  steps_per_epoch=steps_per_epoch,
                  steps_per_eval=steps_per_eval)


def make_train_eval_datasets(distribution_strategy,
                             dataset_name,
                             batch_size,
                             train_augmentation,
                             dataset_kwargs,
                             eval_batch_size=None):
  """Creates Datasets object with training and eval datasets.

  Args:
    distribution_strategy: distribution strategy.
    dataset_name: name of the dataset.
    batch_size: total batch size across all workers.
    train_augmentation: optional callable which performs augmentation of
      training data.
    dataset_kwargs: dictionary with additional arguments which are passed to the
      dataset init function.
    eval_batch_size: optional batch size for evaluation data.

  Returns:
    Datasets namestuple with train and eval datasets.
  """
  if not eval_batch_size:
    eval_batch_size = batch_size

  # If there is more than 1 worker then we need to use
  # experimental_distribute_datasets_from_function, so separate input pipeline
  # is created for each worker / TPU host device.
  # Using separate code branch for TPU donut / single worker device
  # because experimental_distribute_datasets_from_function seems to be
  # much slower than experimental_distribute_dataset in that case.
  num_workers = _num_workers(distribution_strategy)
  per_worker_batch_size = int(batch_size / num_workers)
  per_worker_eval_batch_size = int(eval_batch_size / num_workers)
  logging.info('Distributing input pipeline across %d workers '
               'with %d train and %d eval per worker batch size',
               num_workers, per_worker_batch_size, per_worker_eval_batch_size)

  dataset_name = dataset_name.lower()
  if dataset_name == 'imagenet':
    train_dataset_object = imagenet.ImageNetInput(
        split='train',
        is_training=True,
        batch_size=per_worker_batch_size,
        augmentation=train_augmentation,
        **dataset_kwargs)
    eval_dataset_object = imagenet.ImageNetInput(
        split='test',
        is_training=False,
        batch_size=per_worker_eval_batch_size,
        **dataset_kwargs)
  else:
    raise ValueError('Unsupported dataset name: {0}'.format(dataset_name))

  return _make_datasets_namedtuple(distribution_strategy,
                                   train_dataset_object,
                                   eval_dataset_object)


def make_ssl_train_eval_datasets(distribution_strategy,
                                 dataset_name,
                                 supervised_batch_size,
                                 unsupervised_batch_size,
                                 supervised_train_augmentation,
                                 unsupervised_train_augmentation,
                                 dataset_kwargs,
                                 eval_batch_size=None):
  """Creates a Datasets object for semi-supervised training task.

  Args:
    distribution_strategy: distribution strategy.
    dataset_name: name of the dataset.
    supervised_batch_size: total batch size for supervised data.
    unsupervised_batch_size: total batch size for unsupervised data.
    supervised_train_augmentation: optional callable which performs augmentation
      of supervised training data.
    unsupervised_train_augmentation: optional callable which performs
      augmentation of unsupervised training data.
    dataset_kwargs: dictionary with additional arguments which are passed to the
      dataset init function.
    eval_batch_size: optional batch size for evaluation data.

  Returns:
    Datasets namestuple with train and eval datasets.
  """
  if not eval_batch_size:
    eval_batch_size = max(supervised_batch_size, unsupervised_batch_size)

  # If there is more than 1 worker then we need to use
  # experimental_distribute_datasets_from_function, so separate input pipeline
  # is created for each worker / TPU host device.
  # Using separate code branch for TPU donut / single worker device
  # because experimental_distribute_datasets_from_function seems to be
  # much slower than experimental_distribute_dataset in that case.
  num_workers = _num_workers(distribution_strategy)
  per_worker_sup_batch_size = int(supervised_batch_size / num_workers)
  per_worker_unsup_batch_size = int(unsupervised_batch_size / num_workers)
  per_worker_eval_batch_size = int(eval_batch_size / num_workers)
  logging.info('Distributing input pipeline across %d workers '
               'with %d supervised, %d unsupervised and %d eval '
               'per worker batch size',
               num_workers,
               per_worker_sup_batch_size,
               per_worker_unsup_batch_size,
               per_worker_eval_batch_size)

  dataset_name = dataset_name.lower()
  if dataset_name.startswith('imagenet'):
    supervised_split = 'train' + dataset_name[len('imagenet'):]
    train_dataset_object = None
    train_dataset_object = imagenet.ImageNetSslTrainInput(
        supervised_split=supervised_split,
        supervised_batch_size=per_worker_sup_batch_size,
        unsupervised_batch_size=per_worker_unsup_batch_size,
        supervised_augmentation=supervised_train_augmentation,
        unsupervised_augmentation=unsupervised_train_augmentation,
        **dataset_kwargs)
    eval_dataset_object = imagenet.ImageNetInput(
        split='test',
        is_training=False,
        batch_size=per_worker_eval_batch_size,
        **dataset_kwargs)
  else:
    raise ValueError('Unsupported dataset name: {0}'.format(dataset_name))

  return _make_datasets_namedtuple(distribution_strategy,
                                   train_dataset_object,
                                   eval_dataset_object)
