#!/usr/bin/env python3
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
"""Report all 'stats/accuracy.json' into a json file on stdout.

All the accuracies are summarized.
"""
import json
import sys
import threading

import tensorflow as tf
import tqdm
from absl import app
from absl import flags

FLAGS = flags.FLAGS
N_THREADS = 100


def add_contents_to_dict(filename: str, target):
    with tf.gfile.Open(filename, 'r') as f:
        target[filename] = json.load(f)


def main(argv):
    files = []
    for path in argv[1:]:
        files.extend(tf.io.gfile.glob(path))
    assert files, 'No files found'
    print('Found %d files.' % len(files), file=sys.stderr)
    summary = {}
    threads = []
    for x in tqdm.tqdm(files, leave=False, desc='Collating'):
        t = threading.Thread(
            target=add_contents_to_dict, kwargs=dict(filename=x, target=summary))
        threads.append(t)
        t.start()
        while len(threads) >= N_THREADS:
            dead = [p for p, t in enumerate(threads) if not t.is_alive()]
            while dead:
                p = dead.pop()
                del threads[p]
        if x == files[-1]:
            for t in threads:
                t.join()

    assert len(summary) == len(files)
    print(json.dumps(summary, sort_keys=True, indent=4))


if __name__ == '__main__':
    app.run(main)
