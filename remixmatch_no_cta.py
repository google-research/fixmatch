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

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import data, layers, utils
from libml.utils import EasyDict
from mixmatch import MixMatch

FLAGS = flags.FLAGS


class ReMixMatch(MixMatch):

    def classifier_rot(self, x):
        with tf.variable_scope('classify_rot', reuse=tf.AUTO_REUSE):
            return tf.layers.dense(x, 4, kernel_initializer=tf.glorot_normal_initializer())

    def guess_label(self, logits_y, p_data, p_model, T, use_dm, redux, **kwargs):
        del kwargs
        if redux == 'swap':
            p_model_y = tf.concat([tf.nn.softmax(x) for x in logits_y[1:] + logits_y[:1]], axis=0)
        elif redux == 'mean':
            p_model_y = sum(tf.nn.softmax(x) for x in logits_y) / len(logits_y)
            p_model_y = tf.tile(p_model_y, [len(logits_y), 1])
        elif redux == '1st':
            p_model_y = tf.nn.softmax(logits_y[0])
            p_model_y = tf.tile(p_model_y, [len(logits_y), 1])
        else:
            raise NotImplementedError()

        # Compute the target distribution.
        # 1. Rectify the distribution or not.
        if use_dm:
            p_ratio = (1e-6 + p_data) / (1e-6 + p_model)
            p_weighted = p_model_y * p_ratio
            p_weighted /= tf.reduce_sum(p_weighted, axis=1, keep_dims=True)
        else:
            p_weighted = p_model_y
        # 2. Apply sharpening.
        p_target = tf.pow(p_weighted, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def model(self, batch, lr, wd, beta, w_kl, w_match, w_rot, K, use_xe, warmup_kimg=1024, T=0.5,
              mixmode='xxy.yxy', dbuf=128, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch, K + 1] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')

        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)
        augment = layers.MixMode(mixmode)

        gpu = utils.get_gpu()

        def classifier_to_gpu(x, **kw):
            with tf.device(next(gpu)):
                return self.classifier(x, **kw, **kwargs).logits

        def random_rotate(x):
            b4 = batch // 4
            x, xt = x[:2 * b4], tf.transpose(x[2 * b4:], [0, 2, 1, 3])
            l = np.zeros(b4, np.int32)
            l = tf.constant(np.concatenate([l, l + 1, l + 2, l + 3], axis=0))
            return tf.concat([x[:b4], x[b4:, ::-1, ::-1], xt[:b4, ::-1], xt[b4:, :, ::-1]], axis=0), l

        # Moving average of the current estimated label distribution
        p_model = layers.PMovingAverage('p_model', self.nclass, dbuf)
        p_target = layers.PMovingAverage('p_target', self.nclass, dbuf)  # Rectified distribution (only for plotting)

        # Known (or inferred) true unlabeled distribution
        p_data = layers.PData(self.dataset)

        if w_rot > 0:
            rot_y, rot_l = random_rotate(y_in[:, 1])
            with tf.device(next(gpu)):
                rot_logits = self.classifier_rot(self.classifier(rot_y, training=True, **kwargs).embeds)
            loss_rot = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(rot_l, 4), logits=rot_logits)
            loss_rot = tf.reduce_mean(loss_rot)
            tf.summary.scalar('losses/rot', loss_rot)
        else:
            loss_rot = 0

        if kwargs['redux'] == '1st' and w_kl <= 0:
            logits_y = [classifier_to_gpu(y_in[:, 0], training=True)] * (K + 1)
        elif kwargs['redux'] == '1st':
            logits_y = [classifier_to_gpu(y_in[:, i], training=True) for i in range(2)]
            logits_y += logits_y[:1] * (K - 1)
        else:
            logits_y = [classifier_to_gpu(y_in[:, i], training=True) for i in range(K + 1)]

        guess = self.guess_label(logits_y, p_data(), p_model(), T=T, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        if w_kl > 0:
            w_kl *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
            loss_kl = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ly[:batch], logits=logits_y[1])
            loss_kl = tf.reduce_mean(loss_kl)
            tf.summary.scalar('losses/kl', loss_kl)
        else:
            loss_kl = 0
        del logits_y

        lx = tf.one_hot(l_in, self.nclass)
        xy, labels_xy = augment([xt_in] + [y_in[:, i] for i in range(K + 1)], [lx] + tf.split(ly, K + 1),
                                [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        del xy, labels_xy

        batches = layers.interleave([x] + y, batch)
        logits = [classifier_to_gpu(yi, training=True) for yi in batches[:-1]]
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits.append(classifier_to_gpu(batches[-1], training=True))
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)
        del batches, logits

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        if use_xe:
            loss_xeu = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_y, logits=logits_y)
        else:
            loss_xeu = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_xeu = tf.reduce_mean(loss_xeu)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/%s' % ('xeu' if use_xe else 'l2u'), loss_xeu)
        self.distribution_summary(p_data(), p_model(), p_target())

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.extend([ema_op,
                         p_model.update(guess.p_model),
                         p_target.update(guess.p_target)])
        if p_data.has_update:
            post_ops.append(p_data.update(lx))

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + w_kl * loss_kl + w_match * loss_xeu + w_rot * loss_rot + wd * loss_wd,
            colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_op=tf.nn.softmax(classifier_to_gpu(x_in, getter=ema_getter, training=False)),
            classify_raw=tf.nn.softmax(classifier_to_gpu(x_in, training=False)))  # No EMA, for debugging.


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.MANY_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = ReMixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,

        K=FLAGS.K,
        beta=FLAGS.beta,
        w_kl=FLAGS.w_kl,
        w_match=FLAGS.w_match,
        w_rot=FLAGS.w_rot,
        redux=FLAGS.redux,
        use_dm=FLAGS.use_dm,
        use_xe=FLAGS.use_xe,
        warmup_kimg=FLAGS.warmup_kimg,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
    flags.DEFINE_float('w_kl', 0.5, 'Weight for KL loss.')
    flags.DEFINE_float('w_match', 1.5, 'Weight for distribution matching loss.')
    flags.DEFINE_float('w_rot', 0.5, 'Weight for rotation loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('warmup_kimg', 1024, 'Unannealing duration for SSL loss.')
    flags.DEFINE_enum('redux', 'swap', 'swap mean 1st'.split(), 'Logit selection.')
    flags.DEFINE_bool('use_dm', True, 'Whether to use distribution matching.')
    flags.DEFINE_bool('use_xe', True, 'Whether to use cross-entropy or Brier.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
