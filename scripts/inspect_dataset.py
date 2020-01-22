#!/usr/bin/env python

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

"""Script to inspect a dataset, in particular label distribution.
"""

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from libml import data, utils

flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('samples', 1 << 16, 'Number of samples to load.')

FLAGS = flags.FLAGS


def main(argv):
    utils.setup_main()
    del argv
    utils.setup_tf()
    nbatch = FLAGS.samples // FLAGS.batch
    dataset = data.DATASETS()[FLAGS.dataset]()
    groups = [('labeled', dataset.train_labeled),
              ('unlabeled', dataset.train_unlabeled),
              ('test', dataset.test.repeat())]
    groups = [(name, ds.batch(FLAGS.batch).prefetch(16).make_one_shot_iterator().get_next())
              for name, ds in groups]
    with tf.train.MonitoredSession() as sess:
        for group, train_data in groups:
            stats = np.zeros(dataset.nclass, np.int32)
            minmax = [], []
            for _ in trange(nbatch, leave=False, unit='img', unit_scale=FLAGS.batch, desc=group):
                v = sess.run(train_data)['label']
                for u in v:
                    stats[u] += 1
                minmax[0].append(v.min())
                minmax[1].append(v.max())
            print(group)
            print('  Label range', min(minmax[0]), max(minmax[1]))
            print('  Stats', ' '.join(['%.2f' % (100 * x) for x in (stats / stats.max())]))


if __name__ == '__main__':
    app.run(main)
