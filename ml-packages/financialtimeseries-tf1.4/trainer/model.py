# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Define a Wide + Deep model for classification on structured data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import six
import tensorflow as tf


HEADER = [
  'snp_label', 'snp_m1', 'snp_m2', 'nyse_m1', 'nyse_m2',
  'djia_m1', 'djia_m2', 'nikkei_m0', 'nikkei_m1',
  'hangseng_m0', 'hangseng_m1', 'ftse_m0', 'ftse_m1',
  'dax_m0', 'dax_m1', 'aord_m0', 'aord_m1'
]
TARGET_NAME = 'snp_label'
TARGET_VALUES = ['+', '-']
DEFAULTS = [
  ['null'], [0.0], [0.0], [0.0], [0.0],
  [0.0], [0.0], [0.0], [0.0],
  [0.0], [0.0], [0.0], [0.0],
  [0.0], [0.0], [0.0], [0.0]
]

def parse_csv_row(csv_row):
  columns = tf.decode_csv(tf.expand_dims(csv_row, -1), record_defaults=DEFAULTS)
  features = dict(zip(HEADER, columns))
  target = features.pop(TARGET_NAME)
  return features, target

# to be applied in traing and serving
def process_features(features):
    return features


def csv_input_fn(file_name, mode=tf.estimator.ModeKeys.EVAL,
                 skip_header_lines=0,
                 num_epochs=1,
                 batch_size=500):

  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

  print(file_name)
  file_names = tf.matching_files(file_name)
  
  dataset = tf.data.TextLineDataset(filenames=file_names)
  dataset = dataset.skip(skip_header_lines)

  if shuffle:
      dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row))
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()

  features, target = iterator.get_next()
  return features, target

def get_deep_and_wide_columns():
  snp_m1=tf.feature_column.numeric_column('snp_m1')
  snp_m2=tf.feature_column.numeric_column('snp_m2')
  nyse_m1=tf.feature_column.numeric_column('nyse_m1')
  nyse_m2=tf.feature_column.numeric_column('nyse_m2')
  djia_m1=tf.feature_column.numeric_column('djia_m1')
  djia_m2=tf.feature_column.numeric_column('djia_m2')
  nikkei_m0=tf.feature_column.numeric_column('nikkei_m0')
  nikkei_m1=tf.feature_column.numeric_column('nikkei_m1')
  hangseng_m0=tf.feature_column.numeric_column('hangseng_m0')
  hangseng_m1=tf.feature_column.numeric_column('hangseng_m1')
  ftse_m0=tf.feature_column.numeric_column('ftse_m0')
  ftse_m1=tf.feature_column.numeric_column('ftse_m1')
  dax_m0=tf.feature_column.numeric_column('dax_m0')
  dax_m1=tf.feature_column.numeric_column('dax_m1')
  aord_m0=tf.feature_column.numeric_column('aord_m0')
  aord_m1=tf.feature_column.numeric_column('aord_m1')

  snp_m1_bucketized = tf.feature_column.bucketized_column(
      snp_m1,
      boundaries=[-.05, -.04, -.03, -.02, -.01, .00, .01, .02, .03, 0.4, 0.5]
  )

  nyse_m1_bucketized = tf.feature_column.bucketized_column(
      nyse_m1,
      boundaries=[-.05, -.04, -.03, -.02, -.01, .00, .01, .02, .03, 0.4, 0.5]
  )

  snp_m1_x_nyse_m1 = tf.feature_column.crossed_column(
      [snp_m1_bucketized, nyse_m1_bucketized],
      hash_bucket_size=10
  )

  wide_columns = [snp_m1_bucketized, nyse_m1_bucketized, snp_m1_x_nyse_m1]

  deep_columns = [
     snp_m1, snp_m2, nyse_m1, nyse_m2,
    djia_m1, djia_m2, nikkei_m0, nikkei_m1,
    hangseng_m0, hangseng_m1, ftse_m0, ftse_m1,
    dax_m0, dax_m1, aord_m0, aord_m1
  ]

  return wide_columns, deep_columns


def create_DNNLinearCombinedClassifier(run_config, hparams):

    wide_columns, deep_columns = get_deep_and_wide_columns()

    dnn_optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=hparams.hidden_units,
        label_vocabulary=TARGET_VALUES,
        config = run_config
                )

    return estimator

def csv_serving_input_fn():

  SERVING_HEADER = [
    'snp_m1', 'snp_m2', 'nyse_m1', 'nyse_m2',
    'djia_m1', 'djia_m2', 'nikkei_m0', 'nikkei_m1',
    'hangseng_m0', 'hangseng_m1', 'ftse_m0', 'ftse_m1',
    'dax_m0', 'dax_m1', 'aord_m0', 'aord_m1'
  ]
  SERVING_HEADER_DEFAULTS = [
    [0.0], [0.0], [0.0], [0.0],
    [0.0], [0.0], [0.0], [0.0],
    [0.0], [0.0], [0.0], [0.0],
    [0.0], [0.0], [0.0], [0.0]
  ]

  rows_string_tensor = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='csv_rows')

  receiver_tensor = {'csv_rows': rows_string_tensor}

  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=SERVING_HEADER_DEFAULTS)
  features = dict(zip(SERVING_HEADER, columns))

  # apply feature preprocessing used input_fn
  features = process_features(features)

  return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)

def json_serving_input_fn():

  SERVING_FIELDS= [
    'snp_m1', 'snp_m2', 'nyse_m1', 'nyse_m2',
    'djia_m1', 'djia_m2', 'nikkei_m0', 'nikkei_m1',
    'hangseng_m0', 'hangseng_m1', 'ftse_m0', 'ftse_m1',
    'dax_m0', 'dax_m1', 'aord_m0', 'aord_m1'
  ]

  # apply feature preprocessing used input_fn

  features = {}
  for fiels in SERVING_FIELDS:
    features[fiels] = tf.placeholder(shape=[None], dtype=tf.float32)
  
  # apply feature preprocessing used input_fn
  features = process_features(features)

  return tf.estimator.export.ServingInputReceiver(
        features, features)
