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
"""MixMatch training.
- Ensure class consistency by producing a group of `nu` augmentations of the same image and guessing the label for the
  group.
- Sharpen the target distribution.
- Use the sharpened distribution directly as a smooth label in MixUp.
"""

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import layers, utils, models
from libml.data import PAIR_DATASETS
from libml.layers import MixMode
from libml.utils import EasyDict

FLAGS = flags.FLAGS


class MixMatch(models.MultiModel):

    def distribution_summary(self, p_data, p_model, p_target=None):
        def kl(p, q):
            p /= tf.reduce_sum(p)
            q /= tf.reduce_sum(q)
            return -tf.reduce_sum(p * tf.log(q / p))

        tf.summary.scalar('metrics/kld', kl(p_data, p_model))
        if p_target is not None:
            tf.summary.scalar('metrics/kld_target', kl(p_data, p_target))

        for i in range(self.nclass):
            tf.summary.scalar('matching/class%d_ratio' % i, p_model[i] / p_data[i])
        for i in range(self.nclass):
            tf.summary.scalar('matching/val%d' % i, p_model[i])

    def augment(self, x, l, beta, **kwargs):
        assert 0, 'Do not call.'

    def guess_label(self, y, classifier, T, **kwargs):
        del kwargs
        logits_y = [classifier(yi, training=True) for yi in y]
        logits_y = tf.concat(logits_y, 0)
        # Compute predicted probability distribution py.
        p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)
        # Compute the target distribution.
        p_target = tf.pow(p_model_y, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, nu=2, mixmode='xxy.yxy', dbuf=128, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch, nu] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')

        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)
        augment = MixMode(mixmode)
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits

        # Moving average of the current estimated label distribution
        p_model = layers.PMovingAverage('p_model', self.nclass, dbuf)
        p_target = layers.PMovingAverage('p_target', self.nclass, dbuf)  # Rectified distribution (only for plotting)

        # Known (or inferred) true unlabeled distribution
        p_data = layers.PData(self.dataset)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        lx = tf.one_hot(l_in, self.nclass)
        xy, labels_xy = augment([xt_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        del xy, labels_xy

        batches = layers.interleave([x] + y, batch)
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = [classifier(batches[0], training=True)]
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        for batchi in batches[1:]:
            logits.append(classifier(batchi, training=True))
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_l2u = tf.reduce_mean(loss_l2u)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/l2u', loss_l2u)
        self.distribution_summary(p_data(), p_model(), p_target())

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + w_match * loss_l2u + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,

        beta=FLAGS.beta,
        w_match=FLAGS.w_match,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
