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
"""Interpolation Consistency Training for Semi-Supervised Learning.

Reimplementation of https://arxiv.org/abs/1903.03825
"""

import functools
import os

import tensorflow as tf
from absl import app
from absl import flags

from libml import models, utils
from libml.data import PAIR_DATASETS
from libml.utils import EasyDict

FLAGS = flags.FLAGS


class ICT(models.MultiModel):

    def model(self, batch, lr, wd, ema, warmup_pos, consistency_weight, beta, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch, 2] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')
        l = tf.one_hot(l_in, self.nclass)
        wd *= lr
        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        y_1, y_2 = tf.split(y, 2)

        mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x_in)[0], 1, 1, 1])
        mix = tf.maximum(mix, 1 - mix)

        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        logits_x = classifier(xt_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        logits_teacher = classifier(y_1, training=True, getter=ema_getter)
        labels_teacher = tf.stop_gradient(tf.nn.softmax(logits_teacher))
        labels_teacher = labels_teacher * mix[:, :, 0, 0] + labels_teacher[::-1] * (1 - mix[:, :, 0, 0])
        logits_student = classifier(y_1 * mix + y_1[::-1] * (1 - mix), training=True)
        loss_mt = tf.reduce_mean((labels_teacher - tf.nn.softmax(logits_student)) ** 2, -1)
        loss_mt = tf.reduce_mean(loss_mt)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/mt', loss_mt)

        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss + loss_mt * warmup * consistency_weight,
                                                       colocate_gradients_with_ops=True)
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
    model = ICT(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        warmup_pos=FLAGS.warmup_pos,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,
        consistency_weight=FLAGS.consistency_weight,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('consistency_weight', 50., 'Consistency weight.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
