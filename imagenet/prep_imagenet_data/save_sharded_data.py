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
"""Saves TFRecords as sharded data with given order.
"""

import os
import time

from absl import app
from absl import flags

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Directory with input data.')

flags.DEFINE_string('sharding_file', None,
                    'File which stores information about desired sharding')

flags.DEFINE_string('output_file_prefix', None, 'Prefix of the output file')

flags.DEFINE_integer('num_shards', 0,
                     'Total number of output shards')

flags.DEFINE_integer('max_records_to_process', -1,
                     'Maximum number of records to red from input files.')

flags.DEFINE_boolean('fail_on_missing_examples', False,
                     'If true then pipeline will fail if any input example is '
                     'missing from sharding data.')


def read_sharding_data():
  with tf.io.gfile.GFile(FLAGS.sharding_file) as f:
    lines = f.readlines()
  lines = [line.strip().split(',') for line in lines]
  return {line[0]: (int(line[1]), int(line[2])) for line in lines}


class AugmentExampleWithShardAndNewLabelFn(beam.DoFn):
  """DoFn which augments examples with shard id and index within shard."""

  def process(self, ex, sharding_data):
    filename = ex.features.feature['image/filename'].bytes_list.value[0]
    filename = filename.decode('utf-8')  # decoding python3 bytes into string
    if filename in sharding_data:
      shard_id, idx = sharding_data[filename]
      yield shard_id, (idx, ex)
    elif FLAGS.fail_on_missing_examples:
      raise KeyError('Example not found: {0}'.format(filename))


def save_grouped_data(inputs):
  shard_id, elements = inputs
  output_filename = (FLAGS.output_file_prefix
                     + '-{0:05}-of-{1:05}'.format(shard_id, FLAGS.num_shards))
  with tf.io.TFRecordWriter(output_filename) as writer:
    for _, ex in sorted(elements):
      writer.write(ex.SerializeToString())


def save_sharded_data_pipeline(p):
  """Constructs pipeline which save data into shards."""
  # Read data
  p = p | beam.io.ReadFromTFRecord(
      os.path.join(FLAGS.input_dir, 'train-*'),
      coder=beam.coders.ProtoCoder(tf.train.Example))
  if FLAGS.max_records_to_process > 0:
    p = p | beam.transforms.combiners.Sample.FixedSizeGlobally(
        FLAGS.max_records_to_process)
    p = p | beam.FlatMap(lambda x: x)

  # Regrouping data into shards
  p = p | beam.ParDo(AugmentExampleWithShardAndNewLabelFn(),
                     read_sharding_data())
  p = p | beam.GroupByKey()

  # Save data
  _ = p | beam.Map(save_grouped_data)


def main(argv):
  pipeline_options = PipelineOptions(argv[1:])
  p = beam.Pipeline(options=pipeline_options)
  save_sharded_data_pipeline(p)
  start_time = time.time()
  p.run().wait_until_finish()
  print('Processing done, execition took ', int(time.time() - start_time), ' seconds.')


if __name__ == '__main__':
  app.run(main)
