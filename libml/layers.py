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
"""Custom neural network layers and primitives.
"""
import numbers

import numpy as np
import tensorflow as tf

from libml.data import DataSets


def smart_shape(x):
    s, t = x.shape, tf.shape(x)
    return [t[i] if s[i].value is None else s[i] for i in range(len(s))]


def entropy_from_logits(logits):
    """Computes entropy from classifier logits.

    Args:
        logits: a tensor of shape (batch_size, class_count) representing the
        logits of a classifier.

    Returns:
        A tensor of shape (batch_size,) of floats giving the entropies
        batchwise.
    """
    distribution = tf.contrib.distributions.Categorical(logits=logits)
    return distribution.entropy()


def entropy_penalty(logits, entropy_penalty_multiplier, mask):
    """Computes an entropy penalty using the classifier logits.

    Args:
        logits: a tensor of shape (batch_size, class_count) representing the
            logits of a classifier.
        entropy_penalty_multiplier: A float by which the entropy is multiplied.
        mask: A tensor that optionally masks out some of the costs.

    Returns:
        The mean entropy penalty
    """
    entropy = entropy_from_logits(logits)
    losses = entropy * entropy_penalty_multiplier
    losses *= tf.cast(mask, tf.float32)
    return tf.reduce_mean(losses)


def kl_divergence_from_logits(logits_a, logits_b):
    """Gets KL divergence from logits parameterizing categorical distributions.

    Args:
        logits_a: A tensor of logits parameterizing the first distribution.
        logits_b: A tensor of logits parameterizing the second distribution.

    Returns:
        The (batch_size,) shaped tensor of KL divergences.
    """
    distribution1 = tf.contrib.distributions.Categorical(logits=logits_a)
    distribution2 = tf.contrib.distributions.Categorical(logits=logits_b)
    return tf.contrib.distributions.kl_divergence(distribution1, distribution2)


def mse_from_logits(output_logits, target_logits):
    """Computes MSE between predictions associated with logits.

    Args:
        output_logits: A tensor of logits from the primary model.
        target_logits: A tensor of logits from the secondary model.

    Returns:
        The mean MSE
    """
    diffs = tf.nn.softmax(output_logits) - tf.nn.softmax(target_logits)
    squared_diffs = tf.square(diffs)
    return tf.reduce_mean(squared_diffs, -1)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]


def renorm(v):
    return v / tf.reduce_sum(v, axis=-1, keepdims=True)


def shakeshake(a, b, training):
    if not training:
        return 0.5 * (a + b)
    mu = tf.random_uniform([tf.shape(a)[0]] + [1] * (len(a.shape) - 1), 0, 1)
    mixf = a + mu * (b - a)
    mixb = a + mu[::1] * (b - a)
    return tf.stop_gradient(mixf - mixb) + mixb


class PMovingAverage:
    def __init__(self, name, nclass, buf_size):
        # MEAN aggregation is used by DistributionStrategy to aggregate
        # variable updates across shards
        self.ma = tf.Variable(tf.ones([buf_size, nclass]) / nclass,
                              trainable=False,
                              name=name,
                              aggregation=tf.VariableAggregation.MEAN)

    def __call__(self):
        v = tf.reduce_mean(self.ma, axis=0)
        return v / tf.reduce_sum(v)

    def update(self, entry):
        entry = tf.reduce_mean(entry, axis=0)
        return tf.assign(self.ma, tf.concat([self.ma[1:], [entry]], axis=0))


class PData:
    def __init__(self, dataset: DataSets):
        self.has_update = False
        if dataset.p_unlabeled is not None:
            self.p_data = tf.constant(dataset.p_unlabeled, name='p_data')
        elif dataset.p_labeled is not None:
            self.p_data = tf.constant(dataset.p_labeled, name='p_data')
        else:
            # MEAN aggregation is used by DistributionStrategy to aggregate
            # variable updates across shards
            self.p_data = tf.Variable(renorm(tf.ones([dataset.nclass])),
                                      trainable=False,
                                      name='p_data',
                                      aggregation=tf.VariableAggregation.MEAN)
            self.has_update = True

    def __call__(self):
        return self.p_data / tf.reduce_sum(self.p_data)

    def update(self, entry, decay=0.999):
        entry = tf.reduce_mean(entry, axis=0)
        return tf.assign(self.p_data, self.p_data * decay + entry * (1 - decay))


class MixMode:
    # A class for mixing data for various combination of labeled and unlabeled.
    # x = labeled example
    # y = unlabeled example
    # For example "xx.yxy" means: mix x with x, mix y with both x and y.
    MODES = 'xx.yy xxy.yxy xx.yxy xx.yx xx. .yy xxy. .yxy .'.split()

    def __init__(self, mode):
        assert mode in self.MODES
        self.mode = mode

    @staticmethod
    def augment_pair(x0, l0, x1, l1, beta, **kwargs):
        del kwargs
        if isinstance(beta, numbers.Integral) and beta <= 0:
            return x0, l0

        def np_beta(s, beta):  # TF implementation seems unreliable for beta below 0.2
            return np.random.beta(beta, beta, s).astype('f')

        with tf.device('/cpu'):
            mix = tf.py_func(np_beta, [tf.shape(x0)[0], beta], tf.float32)
            mix = tf.reshape(tf.maximum(mix, 1 - mix), [tf.shape(x0)[0], 1, 1, 1])
            index = tf.random_shuffle(tf.range(tf.shape(x0)[0]))
        xs = tf.gather(x1, index)
        ls = tf.gather(l1, index)
        xmix = x0 * mix + xs * (1 - mix)
        lmix = l0 * mix[:, :, 0, 0] + ls * (1 - mix[:, :, 0, 0])
        return xmix, lmix

    @staticmethod
    def augment(x, l, beta, **kwargs):
        return MixMode.augment_pair(x, l, x, l, beta, **kwargs)

    def __call__(self, xl: list, ll: list, betal: list):
        assert len(xl) == len(ll) >= 2
        assert len(betal) == 2
        if self.mode == '.':
            return xl, ll
        elif self.mode == 'xx.':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
            return [mx0] + xl[1:], [ml0] + ll[1:]
        elif self.mode == '.yy':
            mx1, ml1 = self.augment(
                tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
            return (xl[:1] + tf.split(mx1, len(xl) - 1),
                    ll[:1] + tf.split(ml1, len(ll) - 1))
        elif self.mode == 'xx.yy':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
            mx1, ml1 = self.augment(
                tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
            return ([mx0] + tf.split(mx1, len(xl) - 1),
                    [ml0] + tf.split(ml1, len(ll) - 1))
        elif self.mode == 'xxy.':
            mx, ml = self.augment(
                tf.concat(xl, 0), tf.concat(ll, 0),
                sum(betal) / len(betal))
            return (tf.split(mx, len(xl))[:1] + xl[1:],
                    tf.split(ml, len(ll))[:1] + ll[1:])
        elif self.mode == '.yxy':
            mx, ml = self.augment(
                tf.concat(xl, 0), tf.concat(ll, 0),
                sum(betal) / len(betal))
            return (xl[:1] + tf.split(mx, len(xl))[1:],
                    ll[:1] + tf.split(ml, len(ll))[1:])
        elif self.mode == 'xxy.yxy':
            mx, ml = self.augment(
                tf.concat(xl, 0), tf.concat(ll, 0),
                sum(betal) / len(betal))
            return tf.split(mx, len(xl)), tf.split(ml, len(ll))
        elif self.mode == 'xx.yxy':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
            mx1, ml1 = self.augment(tf.concat(xl, 0), tf.concat(ll, 0), betal[1])
            mx1, ml1 = [tf.split(m, len(xl))[1:] for m in (mx1, ml1)]
            return [mx0] + mx1, [ml0] + ml1
        elif self.mode == 'xx.yx':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
            mx1, ml1 = zip(*[
                self.augment_pair(xl[i], ll[i], xl[0], ll[0], betal[1])
                for i in range(1, len(xl))
            ])
            return [mx0] + list(mx1), [ml0] + list(ml1)
        raise NotImplementedError(self.mode)
