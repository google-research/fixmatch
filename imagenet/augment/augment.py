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
"""Helper code which creates augmentations."""

from absl import logging

from augment.ct_augment import CTAugment
from augment.rand_augment import RandAugment
from augment.weak_augment import flip_augmentation
from augment.weak_augment import noop_augmentation


def _get_augmenter_type_and_args(**kwargs):
  """Extracts augmenter type and args from **kwargs dict."""
  augment_type = kwargs['type'].lower()
  augment_args = {}
  for k, v in kwargs.items():
    if k.startswith(augment_type + '_'):
      augment_args[k[len(augment_type)+1:]] = v
  logging.info('Using augmentation %s with parameters %s',
               augment_type, augment_args)
  return augment_type, augment_args


def create_augmenter(**kwargs):
  """Creates augmenter for supervised task based on hyperparameters dict.

  Args:
    **kwargs: dictionary augment_type and augmenter arguments.

  Returns:
    augmenter_state: class representing augmenter state or None for stateless
      augmnenter
    sup_augmenter: callable which performs augmentation of the data
  """
  augment_type, augment_args = _get_augmenter_type_and_args(**kwargs)
  if not augment_type or (augment_type == 'none') or (augment_type == 'noop'):
    return None, noop_augmentation
  elif augment_type == 'horizontal_flip':
    return None, flip_augmentation
  elif augment_type == 'randaugment':
    return None, RandAugment(**augment_args)
  elif augment_type == 'cta':
    augmenter = CTAugment(**augment_args)
    return augmenter, augmenter
  else:
    raise ValueError('Invalid augmentation type {0}'.format(augment_type))


def create_ssl_augmenter(**kwargs):
  """Creates augmenter for semi-supervised task based on hyperparameters dict.

  Args:
    **kwargs: dictionary augment_type and augmenter arguments.

  Returns:
    augmenter_state: class representing augmenter state or None for stateless
      augmnenter
    sup_augmenter: callable which performs augmentation of supervised data
    unsup_augmenter: callable which performs augmentation of unsupervised data
  """
  augment_type, augment_args = _get_augmenter_type_and_args(**kwargs)
  if not augment_type or (augment_type == 'none') or (augment_type == 'noop'):
    return None, noop_augmentation, noop_augmentation
  elif augment_type == 'horizontal_flip':
    return None, flip_augmentation, flip_augmentation
  elif augment_type == 'randaugment':
    base_augmenter = RandAugment(**augment_args)
    return (None,
            noop_augmentation,
            lambda x: base_augmenter(x, aug_image_key='aug_image'))
  elif augment_type == 'cta':
    base_augmenter = CTAugment(**augment_args)
    return (
        base_augmenter,
        lambda x: base_augmenter(x, probe=True, aug_image_key=None),
        lambda x: base_augmenter(x, probe=False, aug_image_key='aug_image'))
  else:
    raise ValueError('Invalid augmentation type {0}'.format(augment_type))
