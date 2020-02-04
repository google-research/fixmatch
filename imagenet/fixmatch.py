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
"""Fixmatch semi-supervised training code."""

from absl import app
from absl import flags
from absl import logging
from easydict import EasyDict

import tensorflow as tf

import training
from augment.augment import create_ssl_augmenter
from datasets import datasets
from models import resnet50_model
from utils import ema
from utils.learning_rate import ConfigurableLearningRateSchedule


FLAGS = flags.FLAGS


DEFAULT_FIXMATCH_HPARAMS = EasyDict({
    'uratio': 5,  # unsup_batch_size:sup_bathc_size ratio
    'fixmatch': EasyDict({
        'wu': 10.0,
        'confidence': 0.7,
    }),
})


class FixmatchExperiment(training.Experiment):
  """Fixmatch training object."""

  def __init__(self, distribution_strategy, hparams):
    super().__init__(distribution_strategy, hparams)
    # adjust learning rate based on batch size
    self.hparams.learning_rate.base_lr *= self.batch_size / 256

  def create_dataset(self):
    """Creates dataset."""
    augmenter_state, sup_augmenter, unsup_augmenter = (
        create_ssl_augmenter(**self.hparams.augment))
    train_eval_datasets = datasets.make_ssl_train_eval_datasets(
        self.strategy,
        FLAGS.dataset,
        supervised_batch_size=self.batch_size,
        unsupervised_batch_size=int(self.batch_size * self.hparams.uratio),
        eval_batch_size=self.eval_batch_size,
        supervised_train_augmentation=sup_augmenter,
        unsupervised_train_augmentation=unsup_augmenter,
        dataset_kwargs={
            'use_bfloat16': self.hparams.bfloat16,
            'saturate_uint8': self.hparams.input.saturate_uint8,
            'scale_and_center': self.hparams.input.scale_and_center,
            'use_default_augment': self.hparams.input.use_default_augment,
        })
    return train_eval_datasets, augmenter_state

  def create_model(self):
    """Creates model."""
    logging.info('Building model')

    if self.hparams.bfloat16:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    # Create network
    model_creator = resnet50_model.ResNet50(
        weight_decay=self.hparams.weight_decay)
    self.model = model_creator.make_model(
        num_classes=self.datasets.num_classes)
    if self.hparams.use_ema:
      self.model_ema = model_creator.make_model(
          num_classes=self.datasets.num_classes)
      ema.assign_ema_vars_from_initial_values(self.model_ema.variables,
                                              self.model.variables)
    else:
      self.model_ema = None

    # Create optimizer
    self.optimizer = tf.keras.optimizers.SGD(
        learning_rate=ConfigurableLearningRateSchedule(
            self.datasets.steps_per_epoch, **self.hparams.learning_rate),
        momentum=0.9,
        nesterov=True)
    logging.info('Finished building model')

    # Create training metrics
    self.training_loss = tf.keras.metrics.Mean(
        'training_loss', dtype=tf.float32)
    self.training_loss_xe = tf.keras.metrics.Mean(
        'training_loss_xe', dtype=tf.float32)
    self.training_loss_xeu = tf.keras.metrics.Mean(
        'training_loss_xeu', dtype=tf.float32)
    self.training_loss_wd = tf.keras.metrics.Mean(
        'training_loss_wd', dtype=tf.float32)
    self.training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)

    # Prepare checkpointed data
    checkpointed_data = {
        'model': self.model,
        'optimizer': self.optimizer,
    }
    if self.hparams.use_ema:
      checkpointed_data['model_ema'] = self.model_ema
    return checkpointed_data

  @tf.function
  def train_step(self, iterator, num_steps_to_run):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica training step function."""
      sup_image = inputs['image']
      sup_label = inputs['label']
      unsup_image = inputs['unsup_image']
      unsup_aug_image = inputs['unsup_aug_image']
      with tf.GradientTape() as tape:
        images = tf.concat([sup_image, unsup_image, unsup_aug_image], axis=0)
        logits = tf.cast(self.model(images, training=True), tf.float32)

        logit_sup, logit_unsup, logit_aug_unsup = tf.split(
            logits,
            [int(sup_image.shape[0]),
             int(unsup_image.shape[0]),
             int(unsup_aug_image.shape[0])])

        # Supervised loss
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sup_label, logits=logit_sup)
        loss_xe = tf.reduce_mean(loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logit_unsup))
        loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(pseudo_labels, axis=1), logits=logit_aug_unsup)
        pseudo_mask = (tf.reduce_max(pseudo_labels, axis=1) >=
                       self.hparams.fixmatch.confidence)
        pseudo_mask = tf.cast(pseudo_mask, tf.float32)
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)

        # Model weights regularization
        loss_wd = tf.reduce_sum(self.model.losses)

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        total_loss = loss_xe + self.hparams.fixmatch.wu * loss_xeu + loss_wd
        scaled_loss = total_loss / self.strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
      if self.hparams.use_ema:
        ema.update_ema_variables(self.model_ema.variables,
                                 self.model.variables,
                                 self.hparams.ema_decay)
      self.training_loss.update_state(total_loss)
      self.training_loss_xe.update_state(loss_xe)
      self.training_loss_xeu.update_state(self.hparams.fixmatch.wu * loss_xeu)
      self.training_loss_wd.update_state(loss_wd)
      self.training_accuracy.update_state(sup_label, logit_sup)
      if self.update_augmenter_state:
        if self.hparams.use_ema and self.hparams.augment.use_ema_probe_image:
          probe_logits = self.model_ema(inputs['probe_image'], training=False)
        else:
          probe_logits = self.model(inputs['probe_image'], training=False)
        probe_logits = tf.cast(probe_logits, tf.float32)
        self.augmenter_state.update(inputs, tf.nn.softmax(probe_logits))

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  def save_train_metrics(self):
    self.save_metrics({
        'train/total_loss': self.training_loss.result(),
        'loss/xe': self.training_loss_xe.result(),
        'loss/xeu': self.training_loss_xeu.result(),
        'loss/wd': self.training_loss_wd.result(),
        'train/accuracy': self.training_accuracy.result(),
        'train/learning_rate': self.optimizer.learning_rate(
            self.optimizer.iterations),
    })
    logging.info('Training loss: %s, accuracy: %s%% at step %d',
                 round(self.training_loss.result().numpy(), 4),
                 round(self.training_accuracy.result().numpy() * 100, 2),
                 self.optimizer.iterations.numpy())
    self.training_loss.reset_states()
    self.training_accuracy.reset_states()

  def get_current_train_step(self):
    """Returns current training step."""
    return self.optimizer.iterations.numpy()


def main(unused_argv):
  experiment = FixmatchExperiment(
      distribution_strategy=training.create_distribution_strategy(),
      hparams=training.get_hparams(DEFAULT_FIXMATCH_HPARAMS))
  experiment.train_and_eval()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
