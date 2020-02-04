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
"""Helper functions for exponential moving average."""

import tensorflow as tf


def assign_ema_vars_from_initial_values(ema_variables, initial_values):
  """Assign EMA variables from initial values."""

  def _assign_one_var_fn(ema_var, value):
    ema_var.assign(value)

  def _assign_all_in_cross_replica_context_fn(strategy, ema_vars, values):
    for ema_var, value in zip(ema_vars, values):
      value = strategy.extended.reduce_to(
          tf.distribute.ReduceOp.MEAN, value, ema_var)
      if ema_var.trainable:
        strategy.extended.update(ema_var, _assign_one_var_fn, args=(value,))
      else:
        _assign_one_var_fn(ema_var, value)

  replica_context = tf.distribute.get_replica_context()
  if replica_context:
    replica_context.merge_call(_assign_all_in_cross_replica_context_fn,
                               args=(ema_variables, initial_values))
  else:
    if tf.distribute.in_cross_replica_context():
      _assign_all_in_cross_replica_context_fn(tf.distribute.get_strategy(),
                                              ema_variables,
                                              initial_values)
    else:
      for ema_var, value in zip(ema_variables, initial_values):
        _assign_one_var_fn(ema_var, value)


def update_ema_variables(ema_variables, new_values, ema_decay):
  """Updates EMA variables."""
  # Update rule is following:
  #   ema_var := ema_var * ema_decay + var * (1 - ema_decay)
  # which is equivalent to:
  #   ema_var -= (1 - ema_decay) * (ema_var - var)

  one_minus_decay = 1.0 - ema_decay

  def _update_one_var_fn(ema_var, value):
    ema_var.assign_sub((ema_var - value) * one_minus_decay)

  def _update_all_in_cross_replica_context_fn(strategy, ema_vars, values):
    for ema_var, value in zip(ema_vars, values):
      value = strategy.extended.reduce_to(
          tf.distribute.ReduceOp.MEAN, value, ema_var)
      if ema_var.trainable:
        strategy.extended.update(ema_var, _update_one_var_fn, args=(value,))
      else:
        _update_one_var_fn(ema_var, value)

  replica_context = tf.distribute.get_replica_context()
  if replica_context:
    replica_context.merge_call(_update_all_in_cross_replica_context_fn,
                               args=(ema_variables, new_values))
  else:
    if tf.distribute.in_cross_replica_context():
      _update_all_in_cross_replica_context_fn(tf.distribute.get_strategy(),
                                              ema_variables,
                                              new_values)
    else:
      for ema_var, value in zip(ema_variables, new_values):
        _update_one_var_fn(ema_var, value)
