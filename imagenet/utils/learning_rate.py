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
"""Utils for learning rate schedule."""

import tensorflow as tf


class ConfigurableLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Configurable learning rate schedule."""

  def __init__(self,
               steps_per_epoch,
               base_lr,
               use_warmup,
               warmup_epochs,
               decay_rate,
               decay_epochs):
    super(ConfigurableLearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.base_lr = base_lr
    self.use_warmup = use_warmup
    self.warmup_epochs = warmup_epochs
    self.decay_rate = decay_rate
    self.decay_epochs = decay_epochs
    if isinstance(self.decay_epochs, (list, tuple)):
      lr_values = [self.base_lr * (self.decay_rate ** k)
                   for k in range(len(self.decay_epochs) + 1)]
      self.lr_schedule_no_warmup = (
          tf.keras.optimizers.schedules.PiecewiseConstantDecay(
              self.decay_epochs, lr_values))
    else:
      self.lr_schedule_no_warmup = (
          tf.keras.optimizers.schedules.ExponentialDecay(
              self.base_lr, self.decay_epochs, self.decay_rate, staircase=True))

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'base_lr': self.base_lr,
        'use_warmup': self.use_warmup,
        'warmup_epochs': self.warmup_epochs,
        'decay_rate': self.decay_rate,
        'decay_epochs': self.decay_epochs,
    }

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    if self.use_warmup:
      return tf.cond(lr_epoch < self.warmup_epochs,
                     lambda: lr_epoch / self.warmup_epochs * self.base_lr,
                     lambda: self.lr_schedule_no_warmup(lr_epoch))
    else:
      return self.lr_schedule_no_warmup(lr_epoch)
