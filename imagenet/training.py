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
"""Base code for training of all experiments."""

import abc
import copy
import json
import math
import os
import time

from absl import flags
from absl import logging
from easydict import EasyDict

import tensorflow as tf


flags.DEFINE_string(
    'tpu',
    None,
    'Name of the TPU to use. If not specified then local GPUs will be used.')
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored.'))
flags.mark_flag_as_required('model_dir')
flags.DEFINE_integer('per_worker_batch_size', 128, 'Batch size per worker.')
flags.DEFINE_integer('per_worker_eval_batch_size', 128,
                     'Eval batch size per worker.')
flags.DEFINE_boolean('save_summaries_with_epoch', True,
                     'If true then summaries will be reported with epoch*1000 '
                     'as x-coordinate, otherwise current step will be used as '
                     'x-coordinate.')
flags.DEFINE_integer(
    'steps_per_run', 1000, 'Number of steps per one run of training op.')
flags.DEFINE_string('hparams', '', 'JSON with list of hyperparameters.')
flags.DEFINE_string('dataset', None, 'Name of the dataset')
flags.mark_flag_as_required('dataset')

FLAGS = flags.FLAGS


metrics = tf.keras.metrics


DEFAULT_COMMON_HPARAMS = EasyDict({
    'bfloat16': True,
    'num_epochs': 300,
    'use_ema': True,
    'ema_decay': 0.999,  # Reasonable default, higher numbers does not work
    'weight_decay': 0.0003,
    'input': EasyDict({
        'saturate_uint8': True,
        'scale_and_center': True,
        'use_default_augment': True,
    }),
    'learning_rate': EasyDict({
        'base_lr': 0.1,
        'use_warmup': True,
        'warmup_epochs': 5,
        'decay_rate': 0.1,
        'decay_epochs': 50,
    }),
    'augment': EasyDict({
        'type': 'noop',
        'use_ema_probe_image': True,
        'randaugment_num_layers': 2,
        'randaugment_magnitude': None,
        'randaugment_prob_to_apply': 0.5,
        'cta_num_layers': 2,
        'cta_confidence_threshold': 0.85,
        'cta_decay': 0.99,
        'cta_epsilon': 0.001,
        'cta_prob_to_apply': 0.5,
        'cta_num_levels': 15,
    }),
})


def get_hparams(default_model_hparams=None):
  """Returns dictionary with all hyperparameters.

  Args:
    default_model_hparams: dictionary with default model-specific
      hyperparameters.

  Returns:
    dictionary with all parsed hyperparameters.

  This function parses value of the --hparams flag as JSON and returns
  dictionary with all hyperparameters. Note that default values of hyperparams
  are takes from DEFAULT_COMMON_HPARAMS constant and from optional
  default_model_hparams argument.
  """
  hparams_str = FLAGS.hparams.strip()
  if not hparams_str.startswith('{'):
    hparams_str = '{ ' + hparams_str + ' }'
  hparams = copy.deepcopy(DEFAULT_COMMON_HPARAMS)

  def _update_dict(dict_to_update, new_values):
    for k, v in new_values.items():
      if isinstance(v, dict) and (k in dict_to_update):
        _update_dict(dict_to_update[k], v)
      else:
        dict_to_update[k] = v

  if default_model_hparams:
    _update_dict(hparams, default_model_hparams)
  _update_dict(hparams, json.loads(hparams_str))
  return hparams


def safe_mean(losses):
  total = tf.reduce_sum(losses)
  num_elements = tf.dtypes.cast(tf.size(losses), dtype=losses.dtype)
  return tf.math.divide_no_nan(total, num_elements)


def create_distribution_strategy():
  """Creates distribution strategy.

  Returns:
    distribution strategy.

  If flag --tpu is set then TPU distribution strategy will be created,
  otherwise mirrored strategy running on local GPUs will be created.
  """
  if FLAGS.tpu:
    logging.info('Use TPU at %s', FLAGS.tpu)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    distribution_strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    logging.info('Using MirroredStrategy on local devices.')
    distribution_strategy = tf.distribute.MirroredStrategy()

  logging.info('Created distribution strategy: %s', distribution_strategy)

  return distribution_strategy


class Experiment(object):
  """Helper class with most training routines."""

  def __init__(self,
               distribution_strategy,
               hparams):
    self.hparams = hparams
    self.strategy = distribution_strategy
    self.model_dir = FLAGS.model_dir
    num_workers = len(distribution_strategy.extended.worker_devices)
    self.batch_size = FLAGS.per_worker_batch_size * num_workers
    self.eval_batch_size = FLAGS.per_worker_eval_batch_size * num_workers
    self.save_hparams()
    logging.info('Saving checkpoints at %s', self.model_dir)
    logging.info('Hyper parameters: %s', self.hparams)

  @abc.abstractmethod
  def create_dataset(self):
    """Creates dataset.

    Returns:
      datasets: datasets.Datasets named tuple.
      augmenter_state: state of the stateful augmenter.
    """
    pass

  @abc.abstractmethod
  def create_model(self):
    """Creates model and everything needed to train it.

    Returns:
      checkpointed_data: dictionary with model data which needs to be saved to
        checkpoints.
    """
    pass

  def create_or_load_checkpoint(self, **kwargs):
    """Creates and maybe loads checkpoint."""
    checkpoint = tf.train.Checkpoint(**kwargs)
    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
    return checkpoint

  def save_hparams(self):
    """Saves hyperparameters as a json file."""
    filename = os.path.join(self.model_dir, 'hparams.json')
    if not tf.io.gfile.exists(filename):
      with tf.io.gfile.GFile(filename, 'w') as f:
        json.dump(self.hparams, f, indent=2)

  @abc.abstractmethod
  def train_step(self, iterator, num_steps_to_run):
    """Training StepFn."""
    pass

  @tf.function
  def test_step(self, iterator, num_steps_to_run):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica evaluation step function."""
      images, labels = inputs['image'], inputs['label']
      logits = self.model(images, training=False)
      logits = tf.cast(logits, tf.float32)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = safe_mean(loss)
      self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)
      self.test_top5_accuracy.update_state(labels, logits)
      if self.hparams.use_ema:
        ema_logits = self.model_ema(images, training=False)
        self.test_ema_accuracy.update_state(labels, ema_logits)
        self.test_ema_top5_accuracy.update_state(labels, ema_logits)

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @abc.abstractmethod
  def save_train_metrics(self):
    """Saves and resets all training metrics."""
    pass

  @abc.abstractmethod
  def get_current_train_step(self):
    """Returns current training step."""
    pass

  def save_metrics(self, metrics_dict):
    """Saves metrics to event file."""
    step = self.get_current_train_step()
    if FLAGS.save_summaries_with_epoch:
      step = int(step / self.datasets.steps_per_epoch * 1000)
    with self.summary_writer.as_default():
      for k, v in metrics_dict.items():
        tf.summary.scalar(k, v, step=step)

  def train_and_eval(self, steps_per_run=None):
    """Runs training loop with periodic evaluation."""
    if not steps_per_run:
      steps_per_run = FLAGS.steps_per_run
    with self.strategy.scope():
      # Create datasets
      self.datasets, self.augmenter_state = self.create_dataset()
      self.update_augmenter_state = self.augmenter_state is not None
      self.total_train_steps = int(math.ceil(
          self.datasets.steps_per_epoch * self.hparams.num_epochs))

      # Create model
      checkpointed_data = self.create_model()

      if self.update_augmenter_state:
        checkpointed_data.update(self.augmenter_state.get_augmenter_state())
      checkpoint = self.create_or_load_checkpoint(**checkpointed_data)

      # Create eval metrics
      self.test_loss = metrics.Mean('test_loss', dtype=tf.float32)
      self.test_accuracy = metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
      self.test_top5_accuracy = metrics.SparseTopKCategoricalAccuracy(
          k=5, name='test_top5_accuracy', dtype=tf.float32)
      self.test_ema_accuracy = metrics.SparseCategoricalAccuracy(
          'test_ema_accuracy', dtype=tf.float32)
      self.test_ema_top5_accuracy = metrics.SparseTopKCategoricalAccuracy(
          k=5, name='test_ema_accuracy', dtype=tf.float32)

      self.summary_writer = tf.summary.create_file_writer(
          os.path.join(self.model_dir, 'summaries'))

      # training loop
      train_iterator = iter(self.datasets.train_dataset)
      steps_per_second = 0.0
      initial_step = self.get_current_train_step()

      for next_step_to_run in range(initial_step,
                                    self.total_train_steps,
                                    steps_per_run):
        if self.update_augmenter_state:
          self.augmenter_state.sync_state_to_data_pipeline()
          augmenter_state = self.augmenter_state.get_augmenter_state()
          augmenter_state = ['{0}:\n{1}'.format(k, v.numpy())
                             for k, v in augmenter_state.items()]
          logging.info('Augmenter state:\n%s', '\n'.join(augmenter_state))

        logging.info('Running steps %d - %d',
                     next_step_to_run + 1,
                     next_step_to_run + steps_per_run)
        start_time = time.time()
        self.train_step(train_iterator, tf.constant(steps_per_run))
        self.save_train_metrics()
        self.save_metrics({
            'train/steps_per_second': steps_per_second,
            'train/cur_epoch': (self.get_current_train_step()
                                / self.datasets.steps_per_epoch),
        })

        test_iterator = iter(self.datasets.eval_dataset)
        self.test_step(test_iterator, tf.constant(self.datasets.steps_per_eval))
        self.save_metrics({
            'test/loss': self.test_loss.result(),
            'test/accuracy': self.test_accuracy.result(),
            'test/accuracy_top5': self.test_top5_accuracy.result(),
            'test/ema_accuracy': self.test_ema_accuracy.result(),
            'test/ema_accuracy_top5': self.test_ema_top5_accuracy.result(),
        })
        logging.info(
            'Test loss: %s, accuracy (top1 / top5): %s%% / %s%%, '
            'ema_accuracy (top1 / top5): %s%% / %s%%',
            round(self.test_loss.result().numpy(), 4),
            round(self.test_accuracy.result().numpy() * 100, 2),
            round(self.test_top5_accuracy.result().numpy() * 100, 2),
            round(self.test_ema_accuracy.result().numpy() * 100, 2),
            round(self.test_ema_top5_accuracy.result().numpy() * 100, 2))
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        self.test_top5_accuracy.reset_states()
        self.test_ema_accuracy.reset_states()
        self.test_ema_top5_accuracy.reset_states()

        checkpoint_name = checkpoint.save(
            os.path.join(self.model_dir, 'checkpoint'))
        logging.info('Saved checkpoint to %s', checkpoint_name)

        end_time = time.time()
        steps_per_second = steps_per_run / (end_time - start_time)
