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
"""Augmentations for images.
"""
import collections
import functools
import itertools
import multiprocessing
import random

import numpy as np
import tensorflow as tf
from absl import flags

from libml import utils, ctaugment
from libml.utils import EasyDict
from third_party.auto_augment import augmentations, policies

FLAGS = flags.FLAGS
POOL = None
POLICIES = EasyDict(cifar10=policies.cifar10_policies(),
                    cifar100=policies.cifar10_policies(),
                    svhn=policies.svhn_policies(),
                    svhn_noextra=policies.svhn_policies())

RANDOM_POLICY_OPS = (
    'Identity', 'AutoContrast', 'Equalize', 'Rotate',
    'Solarize', 'Color', 'Contrast', 'Brightness',
    'Sharpness', 'ShearX', 'TranslateX', 'TranslateY',
    'Posterize', 'ShearY'
)
AUGMENT_ENUM = 'd x m aa aac ra rac'.split() + ['r%d_%d_%d' % (nops, mag, cutout) for nops, mag, cutout in
                                                itertools.product(range(1, 5), range(1, 16), range(0, 100, 25))] + [
                   'rac%d' % (mag) for mag in range(1, 10)]

flags.DEFINE_integer('K', 1, 'Number of strong augmentation for unlabeled data.')
flags.DEFINE_enum('augment', 'd.d',
                  [x + '.' + y for x, y in itertools.product(AUGMENT_ENUM, AUGMENT_ENUM)] +
                  [x + '.' + y + '.' + z for x, y, z in itertools.product(AUGMENT_ENUM, AUGMENT_ENUM, AUGMENT_ENUM)] + [
                      'd.d.d.d', 'd.aac.d.aac', 'd.rac.d.rac'],
                  'Dataset augmentation method (x=identity, m=mirror, d=default, aa=auto-augment, aac=auto-augment+cutout, '
                  'ra=rand-augment, rac=rand-augment+cutout; for rand-augment, magnitude is also randomized'
                  'rxyy=random augment with x ops and magnitude yy),'
                  'first is for labeled data, others are for unlabeled.')


def init_pool():
    global POOL
    if POOL is None:
        para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
        POOL = multiprocessing.Pool(para)


def augment_mirror(x):
    return tf.image.random_flip_left_right(x)


def augment_shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.random_crop(y, tf.shape(x))


def augment_noise(x, std):
    return x + std * tf.random_normal(tf.shape(x), dtype=x.dtype)


def numpy_apply_policy(x, policy):
    return augmentations.apply_policy(policy, x).astype('f')


def stack_augment(augment: list):
    def func(x):
        xl = [augment[i](x) if augment[i] is not None else x for i in range(len(augment))]
        return {k: tf.stack([x[k] for x in xl]) for k in xl[0].keys()}

    return func


class Primitives:
    @staticmethod
    def m():
        return lambda x: augment_mirror(x['image'])

    @staticmethod
    def ms(shift):
        return lambda x: augment_shift(augment_mirror(x['image']), shift)

    @staticmethod
    def s(shift):
        return lambda x: augment_shift(x['image'], shift)


AugmentPair = collections.namedtuple('AugmentPair', 'tf numpy')
PoolEntry = collections.namedtuple('PoolEntry', 'payload batch')


class AugmentPool:
    def __init__(self, get_samples):
        self.get_samples = get_samples

    def __call__(self, *args, **kwargs):
        return self.get_samples()


NOAUGMENT = AugmentPair(tf=lambda x: dict(image=x['image'], label=x['label'], index=x.get('index', -1)),
                        numpy=AugmentPool)


class AugmentPoolAA(AugmentPool):

    def __init__(self, get_samples, policy_group):
        init_pool()
        self.get_samples = get_samples
        self.policy_group = policy_group
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, policies = arglist
        return np.stack([augmentations.apply_policy(policy, y) for y, policy in zip(x, policies)]).astype('f')

    def queue_images(self, batch):
        args = []
        image = batch['image']
        if image.ndim == 4:
            for x in range(image.shape[0]):
                args.append((image[x:x + 1], [random.choice(POLICIES[self.policy_group])]))
        else:
            for x in image[:, 1:]:
                args.append((x, [random.choice(POLICIES[self.policy_group]) for _ in range(x.shape[0])]))
        self.queue.append(PoolEntry(payload=POOL.imap(self.numpy_apply_policies, args), batch=batch))

    def fill_queue(self):
        for _ in range(4):
            self.queue_images(self.get_samples())

    def __call__(self, *args, **kwargs):
        del args, kwargs
        batch = self.get_samples()
        entry = self.queue.pop(0)
        samples = np.stack(list(entry.payload))
        if entry.batch['image'].ndim == 4:
            samples = samples.reshape(entry.batch['image'].shape)
            entry.batch['image'] = samples
        else:
            samples = samples.reshape(entry.batch['image'][:, 1:].shape)
            entry.batch['image'][:, 1:] = samples
        self.queue_images(batch)
        return entry.batch


class AugmentPoolAAC(AugmentPoolAA):

    def __init__(self, get_samples, policy_group):
        init_pool()
        self.get_samples = get_samples
        self.policy_group = policy_group
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, policies = arglist
        return np.stack([augmentations.cutout_numpy(augmentations.apply_policy(policy, y)) for y, policy in
                         zip(x, policies)]).astype('f')


class AugmentPoolRAM(AugmentPoolAA):
    # Randomized magnitude
    def __init__(self, get_samples, nops=2, magnitude=10):
        init_pool()
        self.get_samples = get_samples
        self.nops = nops
        self.magnitude = magnitude
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, policies = arglist
        return np.stack([augmentations.apply_policy(policy, y) for y, policy in zip(x, policies)]).astype('f')

    def queue_images(self, batch):
        args = []
        image = batch['image']
        policy = lambda: [(op, 0.5, np.random.randint(1, self.magnitude))
                          for op in np.random.choice(RANDOM_POLICY_OPS, self.nops)]
        if image.ndim == 4:
            for x in range(image.shape[0]):
                args.append((image[x:x + 1], [policy()]))
        else:
            for x in image[:, 1:]:
                args.append((x, [policy() for _ in range(x.shape[0])]))
        self.queue.append(PoolEntry(payload=POOL.imap(self.numpy_apply_policies, args), batch=batch))


class AugmentPoolRAMC(AugmentPoolRAM):
    # Randomized magnitude (inherited from queue images)
    def __init__(self, get_samples, nops=2, magnitude=10):
        init_pool()
        self.get_samples = get_samples
        self.nops = nops
        self.magnitude = magnitude
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, policies = arglist
        return np.stack([augmentations.cutout_numpy(augmentations.apply_policy(policy, y)) for y, policy in
                         zip(x, policies)]).astype('f')


class AugmentPoolRAMC2(AugmentPoolRAM):
    # Randomized magnitude (inherited from queue images)
    def __init__(self, get_samples, nops=2, magnitude=10):
        init_pool()
        self.get_samples = get_samples
        self.nops = nops
        self.magnitude = magnitude
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, policies = arglist
        return np.stack([augmentations.cutout_numpy(augmentations.apply_policy(policy, y)) for y, policy in
                         zip(x, policies)]).astype('f')

    def queue_images(self, batch):
        args = []
        image = batch['image']
        policy = lambda: [(op, 0.5, np.random.randint(1, self.magnitude))
                          for op in np.random.choice(RANDOM_POLICY_OPS, self.nops)]
        if image.ndim == 4:
            for x in range(image.shape[0]):
                args.append((image[x:x + 1], [policy()]))
        else:
            for x in image[:, :]:
                args.append((x, [policy() for _ in range(x.shape[0])]))
        self.queue.append(PoolEntry(payload=POOL.imap(self.numpy_apply_policies, args), batch=batch))

    def __call__(self, *args, **kwargs):
        del args, kwargs
        batch = self.get_samples()
        entry = self.queue.pop(0)
        samples = np.stack(list(entry.payload))
        if entry.batch['image'].ndim == 4:
            samples = samples.reshape(entry.batch['image'].shape)
            entry.batch['image'] = samples
        else:
            samples = samples.reshape(entry.batch['image'][:, :].shape)
            entry.batch['image'][:, :] = samples
        self.queue_images(batch)
        return entry.batch


class AugmentPoolRA(AugmentPoolAA):
    def __init__(self, get_samples, nops, magnitude, cutout):
        init_pool()
        self.get_samples = get_samples
        self.nops = nops
        self.magnitude = magnitude
        self.size = cutout
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, policies, cutout = arglist
        return np.stack([augmentations.cutout_numpy(augmentations.apply_policy(policy, y),
                                                    size=int(0.01 * cutout * min(y.shape[:2])))
                         for y, policy in zip(x, policies)]).astype('f')

    def queue_images(self, batch):
        args = []
        image = batch['image']
        # Fixed magnitude
        policy = lambda: [(op, 1.0, self.magnitude) for op in np.random.choice(RANDOM_POLICY_OPS, self.nops)]
        if image.ndim == 4:
            for x in range(image.shape[0]):
                args.append((image[x:x + 1], [policy()], self.size))
        else:
            for x in image[:, 1:]:
                args.append((x, [policy() for _ in range(x.shape[0])], self.size))
        self.queue.append(PoolEntry(payload=POOL.imap(self.numpy_apply_policies, args), batch=batch))


class AugmentPoolCTA(AugmentPool):

    def __init__(self, get_samples):
        init_pool()
        self.get_samples = get_samples
        self.queue = []
        self.fill_queue()

    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=ctaugment.apply(x, cta.policy(probe=False)))
        assert not probe
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cta.policy(probe=False)) for y in x[1:]]).astype('f'))

    def queue_images(self):
        batch = self.get_samples()
        args = [(x, batch['cta'], batch['probe']) for x in batch['image']]
        self.queue.append(PoolEntry(payload=POOL.imap(self.numpy_apply_policies, args), batch=batch))

    def fill_queue(self):
        for _ in range(4):
            self.queue_images()

    def __call__(self, *args, **kwargs):
        del args, kwargs
        entry = self.queue.pop(0)
        samples = list(entry.payload)
        entry.batch['image'] = np.stack(x['image'] for x in samples)
        if 'probe' in samples[0]:
            entry.batch['probe'] = np.stack(x['probe'] for x in samples)
            entry.batch['policy'] = [x['policy'] for x in samples]
        self.queue_images()
        return entry.batch


DEFAULT_AUGMENT = EasyDict(
    cifar10=AugmentPair(tf=lambda x: dict(image=Primitives.ms(4)(x), label=x['label'], index=x.get('index', -1)),
                        numpy=AugmentPool),
    cifar100=AugmentPair(tf=lambda x: dict(image=Primitives.ms(4)(x), label=x['label'], index=x.get('index', -1)),
                         numpy=AugmentPool),
    fashion_mnist=AugmentPair(tf=lambda x: dict(image=Primitives.ms(4)(x), label=x['label'], index=x.get('index', -1)),
                              numpy=AugmentPool),
    stl10=AugmentPair(tf=lambda x: dict(image=Primitives.ms(12)(x), label=x['label'], index=x.get('index', -1)),
                      numpy=AugmentPool),
    svhn=AugmentPair(tf=lambda x: dict(image=Primitives.s(4)(x), label=x['label'], index=x.get('index', -1)),
                     numpy=AugmentPool),
    svhn_noextra=AugmentPair(tf=lambda x: dict(image=Primitives.s(4)(x), label=x['label'], index=x.get('index', -1)),
                             numpy=AugmentPool),
)
AUTO_AUGMENT = EasyDict({
    k: AugmentPair(tf=v.tf, numpy=functools.partial(AugmentPoolAA, policy_group=k))
    for k, v in DEFAULT_AUGMENT.items()
})
AUTO_AUGMENT_CUTOUT = EasyDict({
    k: AugmentPair(tf=v.tf, numpy=functools.partial(AugmentPoolAAC, policy_group=k))
    for k, v in DEFAULT_AUGMENT.items()
})
RAND_AUGMENT = EasyDict({
    k: AugmentPair(tf=v.tf, numpy=functools.partial(AugmentPoolRAM, nops=2, magnitude=10))
    for k, v in DEFAULT_AUGMENT.items()
})
RAND_AUGMENT_CUTOUT = EasyDict({
    k: AugmentPair(tf=v.tf, numpy=functools.partial(AugmentPoolRAMC, nops=2, magnitude=10))
    for k, v in DEFAULT_AUGMENT.items()
})


def get_augmentation(dataset: str, augmentation: str):
    if augmentation == 'x':
        return NOAUGMENT
    elif augmentation == 'm':
        return AugmentPair(tf=lambda x: dict(image=Primitives.m()(x), label=x['label'], index=x.get('index', -1)),
                           numpy=AugmentPool)
    elif augmentation == 'd':
        return DEFAULT_AUGMENT[dataset]
    elif augmentation == 'aa':
        return AUTO_AUGMENT[dataset]
    elif augmentation == 'aac':
        return AUTO_AUGMENT_CUTOUT[dataset]
    elif augmentation == 'ra':
        return RAND_AUGMENT[dataset]
    elif augmentation.startswith('rac'):
        mag = 10 if augmentation == 'rac' else int(augmentation[-1])
        return AugmentPair(tf=DEFAULT_AUGMENT[dataset].tf,
                           numpy=functools.partial(AugmentPoolRAMC, nops=2, magnitude=mag))
    elif augmentation[0] == 'r':
        nops, mag, cutout = (int(x) for x in augmentation[1:].split('_'))
        return AugmentPair(tf=DEFAULT_AUGMENT[dataset].tf,
                           numpy=functools.partial(AugmentPoolRA, nops=nops, magnitude=mag, cutout=cutout))
    else:
        raise NotImplementedError(augmentation)


def augment_function(dataset: str):
    augmentations = FLAGS.augment.split('.')
    assert len(augmentations) == 2
    return [get_augmentation(dataset, x) for x in augmentations]


def pair_augment_function(dataset: str):
    augmentations = FLAGS.augment.split('.')
    assert len(augmentations) == 3
    unlabeled = [get_augmentation(dataset, x) for x in augmentations[1:]]
    return [get_augmentation(dataset, augmentations[0]),
            AugmentPair(tf=stack_augment([x.tf for x in unlabeled]), numpy=unlabeled[-1].numpy)]


def quad_augment_function(dataset: str):
    augmentations = FLAGS.augment.split('.')
    assert len(augmentations) == 4
    labeled = [get_augmentation(dataset, x) for x in augmentations[:2]]
    unlabeled = [get_augmentation(dataset, x) for x in augmentations[2:]]
    return [AugmentPair(tf=stack_augment([x.tf for x in labeled]), numpy=labeled[-1].numpy),
            AugmentPair(tf=stack_augment([x.tf for x in unlabeled]), numpy=unlabeled[-1].numpy)]


def many_augment_function(dataset: str):
    augmentations = FLAGS.augment.split('.')
    assert len(augmentations) == 3
    unlabeled = [get_augmentation(dataset, x) for x in (augmentations[1:2] + augmentations[2:] * FLAGS.K)]
    return [get_augmentation(dataset, augmentations[0]),
            AugmentPair(tf=stack_augment([x.tf for x in unlabeled]), numpy=unlabeled[-1].numpy)]
