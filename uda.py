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
"""Unsupervised data augmentation (UDA)

"""

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from libml import models, utils
from libml.data import PAIR_DATASETS

FLAGS = flags.FLAGS


class UDA(models.MultiModel):
    TSA_MODES = 'no exp linear log'.split()

    def tsa_threshold(self, tsa, scale=5, tsa_pos=10, **kwargs):
        del kwargs
        # step ratio will be maxed at (2 ** 14) * (2 ** 10) ~ 16.8M updates
        step_ratio = tf.to_float(self.step) / tf.to_float(min(FLAGS.train_kimg, 1 << 14) << tsa_pos)
        if tsa == 'linear':
            coeff = step_ratio
        elif tsa == 'exp':  # [exp(-5), exp(0)] = [1e-2, 1]
            coeff = tf.exp((step_ratio - 1) * scale)
        elif tsa == 'log':  # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
            coeff = 1 - tf.exp((-step_ratio) * scale)
        elif tsa == 'no':
            coeff = tf.to_float(1.0)
        elif tsa != 'no':
            raise NotImplementedError(tsa)
        coeff = tf.math.minimum(coeff, 1.0)  # bound the coefficient
        p_min = 1. / self.nclass
        return coeff * (1 - p_min) + p_min

    def tsa_loss_mask(self, tsa, logits, labels, tsa_pos, **kwargs):
        thresh = self.tsa_threshold(tsa, tsa_pos=tsa_pos, **kwargs)
        p_class = tf.nn.softmax(logits, axis=-1)
        p_correct = tf.reduce_sum(labels * p_class, axis=-1)
        loss_mask = tf.cast(p_correct <= thresh, tf.float32)  # Ignore confident predictions.
        return tf.stop_gradient(loss_mask)

    @staticmethod
    def confidence_based_masking(logits, p_class=None, thresh=0.9):
        if logits is not None:
            p_class = tf.nn.softmax(logits, axis=-1)
        p_class_max = tf.reduce_max(p_class, axis=-1)
        loss_mask = tf.cast(p_class_max >= thresh, tf.float32)  # Ignore unconfident predictions.
        return tf.stop_gradient(loss_mask)

    @staticmethod
    def softmax_temperature_controlling(logits, T):
        # this is essentially the same as sharpening in mixmatch
        logits = logits / T
        return tf.stop_gradient(logits)

    @staticmethod
    def kl_divergence_from_logits(p_logits, q_logits):
        p = tf.nn.softmax(p_logits)
        log_p = tf.nn.log_softmax(p_logits)
        log_q = tf.nn.log_softmax(q_logits)
        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

    @staticmethod
    def entropy_from_logits(logits):
        log_prob = tf.nn.log_softmax(logits, axis=-1)
        prob = tf.exp(log_prob)
        ent = tf.reduce_sum(-prob * log_prob, axis=-1)
        return ent

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, gen_labeled, gen_unlabeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def model(self, batch, lr, wd, wu, we, confidence, uratio,
              temperature=1.0, tsa='no', tsa_pos=10, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')
        l = tf.one_hot(l_in, self.nclass)

        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio+1)
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # softmax temperature control
        logits_weak_tgt = self.softmax_temperature_controlling(logits_weak, T=temperature)
        # generate confidence mask based on sharpened distribution
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        pseudo_mask = self.confidence_based_masking(logits=None, p_class=pseudo_labels, thresh=confidence)
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        tf.summary.scalar('monitors/conf_weak', tf.reduce_mean(tf.reduce_max(tf.nn.softmax(logits_weak), axis=1)))
        tf.summary.scalar('monitors/conf_strong', tf.reduce_mean(tf.reduce_max(tf.nn.softmax(logits_strong), axis=1)))

        kld = self.kl_divergence_from_logits(logits_weak_tgt, logits_strong)
        entropy = self.entropy_from_logits(logits_weak)
        loss_xeu = tf.reduce_mean(kld * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)
        loss_ent = tf.reduce_mean(entropy)
        tf.summary.scalar('losses/entropy', loss_ent)

        # supervised loss with TSA
        loss_mask = self.tsa_loss_mask(tsa=tsa, logits=logits_x, labels=l, tsa_pos=tsa_pos)
        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss_xe = tf.reduce_sum(loss_xe * loss_mask) / tf.math.maximum(tf.reduce_sum(loss_mask), 1.0)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/mask_sup', tf.reduce_mean(loss_mask))

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + loss_xeu * wu + loss_ent * we + loss_wd * wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = UDA(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        wu=FLAGS.wu,
        we=FLAGS.we,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        temperature=FLAGS.temperature,
        tsa=FLAGS.tsa,
        tsa_pos=FLAGS.tsa_pos,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)  # 1024 epochs


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wu', 1, 'Consistency weight.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('we', 0, 'Entropy minimization weight.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('temperature', 1, 'Softmax sharpening temperature.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('tsa_pos', 8, 'TSA change rate.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    flags.DEFINE_enum('tsa', 'no', UDA.TSA_MODES, 'TSA mode.')
    FLAGS.set_default('augment', 'd.d.rac')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
