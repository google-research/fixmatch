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
"""Control theory augment."""

import tensorflow as tf

from augment import augment_ops


IMAGENET_AUG_OPS = [
    'AutoContrastBlend',
    'Blur',
    'Brightness',
    'Color',
    'Contrast',
    'EqualizeBlend',
    'InvertBlend',
    'Identity',
    'Posterize',
    'Rescale',
    'Rotate',
    'Sharpness',
    'ShearX',
    'ShearY',
    'Smooth',
    'Solarize',
    'SolarizeAdd',
    'TranslateX',
    'TranslateY',
]


def _ignore_level_to_arg(level):
  del level
  return ()


def _identity_level_to_arg(level):
  return (level,)


def _enhance_level_to_arg(level):
  return (level * 1.9 + 0.1,)


def _posterize_level_to_arg(level):
  return (1 + int(level * 7.999),)


def _rotate_level_to_arg(level):
  angle_in_degrees = (2 * level - 1) * 45
  return (angle_in_degrees,)


def _shear_level_to_arg(level):
  shear = (2 * level - 1) * 0.3
  return (shear,)


def _solarize_level_to_arg(level):
  return (int(level * 256),)


def _solarize_add_level_to_arg(level):
  return (int(level * 110),)


def _translate_level_to_arg(level):
  shift_pixels = (2 * level - 1) * 100
  return (shift_pixels,)


LEVEL_TO_ARG = {
    'AutoContrastBlend': _identity_level_to_arg,
    'Blur': _identity_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'EqualizeBlend': _identity_level_to_arg,
    'InvertBlend': _identity_level_to_arg,
    'Identity': _ignore_level_to_arg,
    'Posterize': _posterize_level_to_arg,
    'Rescale': _identity_level_to_arg,
    'Rotate': _rotate_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'Smooth': _identity_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'TranslateX': _translate_level_to_arg,
    'TranslateY': _translate_level_to_arg,
}


def _skip_mirrored_creator(next_creator, *args, **kwargs):
  """Skip mirrored variable creation."""
  kwargs['skip_mirrored_creator'] = True
  return next_creator(*args, **kwargs)


def apply_augmentation_op(image, op_index, op_level, prob_to_apply):
  """Applies one augmentation op to the image."""
  branch_fns = []
  for augment_op_name in IMAGENET_AUG_OPS:
    augment_fn = augment_ops.NAME_TO_FUNC[augment_op_name]
    level_to_args_fn = LEVEL_TO_ARG[augment_op_name]
    def _branch_fn(image=image,
                   augment_fn=augment_fn,
                   level_to_args_fn=level_to_args_fn):
      args = [image] + list(level_to_args_fn(op_level))
      return augment_fn(*args)
    branch_fns.append(_branch_fn)
  aug_image = tf.switch_case(op_index, branch_fns, default=lambda: image)
  if prob_to_apply is not None:
    return tf.cond(
        tf.random.uniform(shape=[], dtype=tf.float32) < prob_to_apply,
        lambda: aug_image,
        lambda: image)
  else:
    return aug_image


class CTAugment(object):
  """Implementation of control theory augment."""

  def __init__(self,
               num_layers=2,
               confidence_threshold=0.85,
               decay=0.99,
               epsilon=0.001,
               prob_to_apply=None,
               num_levels=10):
    """Initialize CT Augment.

    Args:
      num_layers: number of augmentation layers, i.e. how many times to do
        augmentation.
      confidence_threshold: confidence threshold for probabilities
      decay: decay factor for augmentation rates
      epsilon: samll number which is used to avoid numerical instabilities
        while computing probabilities.
      prob_to_apply: probability to apply on each layer.
        If None then always apply.
      num_levels: number of levels for quantization of the magnitude.
    """
    # Augmenter args
    self.num_layers = num_layers
    self.confidence_threshold = float(confidence_threshold)
    self.decay = float(decay)
    self.alpha = 1.0 - self.decay
    self.epsilon = epsilon
    self.num_levels = int(num_levels)
    self.prob_to_apply = prob_to_apply
    # State of the augmenter is defined by rates.
    # To speed up sampling we also keep separate variable for sampling
    # probabilities (log_probs) which are deterministically computed from rates.
    self.state_shape = [len(IMAGENET_AUG_OPS), self.num_levels]
    # rates are updated using assign_add and averaged across all replicas.
    self.rates = tf.Variable(
        tf.ones(self.state_shape, dtype=tf.float32),
        trainable=False,
        name='cta_rates',
        aggregation=tf.VariableAggregation.MEAN,
        synchronization=tf.VariableSynchronization.ON_WRITE)
    # log_probs is deterministically computed from rates and value should
    # be the same on all replicas, thus we use ONLY_FIRST_REPLICA aggregation
    self.probs = tf.Variable(
        tf.ones(self.state_shape, dtype=tf.float32) / self.num_levels,
        trainable=False,
        name='cta_probs',
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        synchronization=tf.VariableSynchronization.ON_WRITE)
    # list of log probs variables for each data pipeline
    self.log_probs = []

  def update(self, tensor_dict, probe_probs):
    """Update augmenter state to classification of probe images."""
    # shape of probe_probs is (batch_size, num_classes)
    op_idx = tensor_dict['probe_op_indices']  # shape=(batch_size, num_layers)
    op_arg = tensor_dict['probe_op_args']  # shape=(batch_size, num_layers)
    label = tf.expand_dims(tensor_dict['label'], 1)  # shape=(batch_size, 1)

    # Compute proximity metric as softmax(model(probe_image))[correct_label]
    # Tile proximity, so its shape will be (batch_size, num_layers)
    proximity = tf.gather(probe_probs, label, axis=1, batch_dims=1)
    proximity = tf.tile(proximity, [1, self.num_layers])
    # Quantize op_arg to obtain levels of the ops.
    # NOTE: computed level should be always less than num_levels,
    #       nevertherless use minimum operation to enforce the range.
    level_idx = tf.cast(op_arg * self.num_levels, tf.int32)
    level_idx = tf.minimum(level_idx, self.num_levels)

    # Update rates.
    # For each (op_index, level_index, proximity) in the list of selected ops
    # update rate using following formula:
    #   rate[op_idx, level_idx] = rate[op_idx, level_idx] * decay
    #                             + proximity * (1 - decay)
    # which is equivalent to:
    #   alpha = 1 - decay
    #   rate[op_idx, level_idx] += (proximity - rate[op_idx, level_idx]) * alpha
    #
    # So update is performed using assign_add operation. If several updates
    # correpond to the same (op_idx, level_idx) then they are averaged.
    op_level_idx = tf.concat([tf.reshape(op_idx, [-1, 1]),
                              tf.reshape(level_idx, [-1, 1])],
                             axis=1)
    flat_proximity = tf.reshape(proximity, [-1])
    sparse_update = (
        (flat_proximity - tf.gather_nd(self.rates, op_level_idx)) * self.alpha)
    # Dense matrix with updates is computed in dense_update_numerator.
    # tf.scatter_nd adds up all updates which correspond to the same index,
    # however we need to compute mean. Thus we compute number of
    # updates corresponding to each index and divide by this number.
    dense_update_numerator = tf.scatter_nd(op_level_idx,
                                           sparse_update,
                                           shape=self.state_shape)
    dense_update_denominator = tf.scatter_nd(op_level_idx,
                                             tf.ones_like(sparse_update),
                                             shape=self.state_shape)
    dense_update_denominator = tf.maximum(dense_update_denominator, 1.0)
    self.rates.assign_add(dense_update_numerator / dense_update_denominator)

    # Convert rates to log probabilities
    probs = tf.maximum(self.rates, self.epsilon)
    probs = probs / tf.reduce_max(probs, axis=1, keepdims=True)
    probs = tf.where(probs < self.confidence_threshold,
                     tf.zeros_like(probs), probs)
    probs = probs + self.epsilon
    probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
    self.probs.assign(probs)

  def sync_state_to_data_pipeline(self):
    log_prob_value = tf.math.log(self.probs)
    for v in self.log_probs:
      v.assign(log_prob_value)

  def get_augmenter_state(self):
    """Returns augmenter state to save in checkpoint or for debugging."""
    return {
        'ct_augment_rates': self.rates,
        'ct_augment_probs': self.probs,
    }

  def _sample_ops_uniformly(self):
    """Uniformly samples sequence of augmentation ops."""
    op_indices = tf.random.uniform(
        shape=[self.num_layers], maxval=len(IMAGENET_AUG_OPS), dtype=tf.int32)
    op_args = tf.random.uniform(shape=[self.num_layers], dtype=tf.float32)
    return op_indices, op_args

  def _sample_ops(self, local_log_prob):
    """Samples sequence of augmentation ops using current probabilities."""
    # choose operations
    op_indices = tf.random.uniform(
        shape=[self.num_layers], maxval=len(IMAGENET_AUG_OPS), dtype=tf.int32)
    # sample arguments for each selected operation
    selected_ops_log_probs = tf.gather(local_log_prob, op_indices, axis=0)
    op_args = tf.random.categorical(selected_ops_log_probs, num_samples=1)
    op_args = tf.cast(tf.squeeze(op_args, axis=1), tf.float32)
    op_args = (op_args + tf.random.uniform([self.num_layers])) / self.num_levels
    return op_indices, op_args

  def _apply_ops(self, image, op_indices, op_args, prob_to_apply=None):
    for idx in range(self.num_layers):
      image = apply_augmentation_op(image,
                                    op_indices[idx],
                                    op_args[idx],
                                    prob_to_apply)
    return image

  def __call__(self, image, probe=True, aug_image_key='image'):
    # creating local variable which will store copy of CTA log probabilities
    with tf.variable_creator_scope(_skip_mirrored_creator):
      local_log_prob = tf.Variable(
          lambda: tf.ones(self.state_shape, dtype=tf.float32),
          trainable=False,
          name='cta_log_probs')
    self.log_probs.append(local_log_prob)

    output_dict = {}
    if probe:
      probe_op_indices, probe_op_args = self._sample_ops_uniformly()
      probe_image = self._apply_ops(image, probe_op_indices, probe_op_args)
      output_dict['probe_op_indices'] = probe_op_indices
      output_dict['probe_op_args'] = probe_op_args
      output_dict['probe_image'] = probe_image

    if aug_image_key is not None:
      op_indices, op_args = self._sample_ops(local_log_prob)
      aug_image = self._apply_ops(image, op_indices, op_args,
                                  prob_to_apply=self.prob_to_apply)
      output_dict[aug_image_key] = aug_image

    if aug_image_key != 'image':
      output_dict['image'] = image

    return output_dict
