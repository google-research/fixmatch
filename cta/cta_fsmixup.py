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

import os

from absl import app
from absl import flags

from cta.lib.train import CTAClassifyFullySupervised
from fully_supervised.fs_mixup import FSMixup
from fully_supervised.lib import data
from libml import utils

FLAGS = flags.FLAGS


class CTAFSMixup(FSMixup, CTAClassifyFullySupervised):
    pass


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = CTAFSMixup(
        os.path.join(FLAGS.train_dir, dataset.name, CTAFSMixup.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,
        dropout=FLAGS.dropout,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.002, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_float('dropout', 0, 'Dropout on embedding layer.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('augment', 'd.d')
    FLAGS.set_default('dataset', 'cifar10-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
