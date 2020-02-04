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
"""ResNet50 model.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
"""

import tensorflow as tf


backend = tf.keras.backend
initializers = tf.keras.initializers
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers


DEFAULT_L2_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_NORM_DECAY = 0.9
DEFAULT_BATCH_NORM_EPSILON = 1e-5


class ResNet50(object):
  """Resnet50 model."""

  def __init__(self,
               weight_decay=DEFAULT_L2_WEIGHT_DECAY,
               bn_decay=DEFAULT_BATCH_NORM_DECAY,
               bn_epsilon=DEFAULT_BATCH_NORM_EPSILON):
    self.weight_decay = weight_decay
    self.bn_decay = bn_decay
    self.bn_epsilon = bn_epsilon

  def _identity_block(self, input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Returns:
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, use_bias=False,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

  def _conv_block(self,
                  input_tensor,
                  kernel_size,
                  filters,
                  stage,
                  block,
                  strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.

    Returns:
        Output tensor for the block.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), use_bias=False, strides=strides,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(
                                 self.weight_decay),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=self.bn_decay,
                                         epsilon=self.bn_epsilon,
                                         name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

  def make_model(self, num_classes):
    """Instantiates the ResNet50 architecture."""
    # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
      input_shape = (3, 224, 224)
      bn_axis = 1
    else:
      input_shape = (224, 224, 3)
      bn_axis = 3

    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=self.bn_decay,
                                  epsilon=self.bn_epsilon,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = self._conv_block(x, 3, [64, 64, 256], stage=2, block='a',
                         strides=(1, 1))
    x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = self._conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = self._conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=regularizers.l2(self.weight_decay),
        bias_regularizer=regularizers.l2(self.weight_decay),
        name='fc1000')(x)

    # Create model.
    return models.Model(img_input, x, name='resnet50')
