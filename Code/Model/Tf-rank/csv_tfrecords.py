# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:25:39 2023

@author: corde
"""
import os
import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd

os.chdir(r"C:\Users\corde\Documents\0_Github\NLP-Learning")

datos = pd.read_csv('Data/Raw/tfrecords/datos_a_corregir.csv').loc[:,['TEXTO_COMPARACION', 'D_EVENTO','RANK']]
datos['RANK'] = datos['RANK'].astype('int')

datos.columns = ['document_tokens', 'query_tokens', 'relevance']

types = {'document_tokens':'int',
         'query_tokens': 'str',
         'relevance':'int'}

values_list = list(types.values())
keys_list = list(types.keys())

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_toexample(row, column_types_list, column_names_list):
  feature_dict = {}
  for i in range(0,len(row)):
    if column_types_list[i] == 'str':
      feature_dict[column_names_list[i]] = _bytes_feature(str.encode(row[i]))
    elif column_types_list[i] == 'int':
      feature_dict[column_names_list[i]] = _int64_feature(row[i])
    elif column_types_list[i] == 'float':
      feature_dict[column_names_list[i]] = _float_feature(row[i])
  example = tf.train.Example(features=tf.train.Features(feature = feature_dict))
  return example

with tf.io.TFRecordWriter(path="Data/Raw/tfrecords/datos.tfrecords") as writer:
  for _, row in datos.iterrows():
    example = convert_toexample(row=row, column_types_list=values_list, column_names_list=keys_list)
    writer.write(example.SerializeToString())

def get_feature_description(types):
  feature_description = {}
  for key, value in types.items():
    if value == 'str':
      feature_description[key] = tf.io.FixedLenFeature([], tf.string)
    elif value == 'int':
      feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
    elif value == 'float':
      feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
  return feature_description

def parse_example(record):
  return tf.io.parse_single_example(record, get_feature_description(types))

datos = tf.data.TFRecordDataset("Data/Raw/tfrecords/datos.tfrecords")
parsed_tfrecords = datos.map(map_func=parse_example)
for tfrecord in parsed_tfrecords:
  print(tfrecord['document_tokens'].numpy())
  
#--Estructura de datos tfrecords--#

#Leer el tfrecords

raw_dataset = tf.data.TFRecordDataset('Data/Raw/tfrecords/datos.tfrecords')

for raw_record in raw_dataset.take(1):
  print(repr(raw_record))
    
datos_tfrecords = tf.data.TFRecordDataset("Data/Raw/tfrecords/datos.tfrecords")
parsed_tfrecords = datos_tfrecords.map(map_func=parse_example)
for tfrecord in parsed_tfrecords:
  print(tfrecord['relevance'].numpy())

#¿Como debe ser la estructura de train.tfrecords?

tmp = tf.data.TFRecordDataset('Data/Raw/tfrecords/train.tfrecords')

for raw_record in tmp.take(1):
  print(repr(raw_record))

# Store the paths to files containing training and test instances.
_TRAIN_DATA_PATH = "/tmp/train.tfrecords"
_TEST_DATA_PATH = "/tmp/test.tfrecords"

# Store the vocabulary path for query and document tokens.
_VOCAB_PATH = "Data/Raw/tfrecords/vocab.txt"
_EMBEDDING_DIMENSION = 20

_PADDING_LABEL = -1
_LABEL_FEATURE = "relevance"

# Parameters to the scoring function.
_BATCH_SIZE = 32

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 50

def context_feature_columns():
  """Returns context feature names to column definitions."""
  sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
      key="query_tokens",
      vocabulary_file=_VOCAB_PATH)
  query_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  return {"query_tokens": query_embedding_column}


def example_feature_columns():
  """Returns the example feature columns."""
  sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
      key="document_tokens",
      vocabulary_file=_VOCAB_PATH)
  document_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  return {"document_tokens": document_embedding_column}

def input_fn(path, num_epochs=None):
  context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
  label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
  example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])
  dataset = tfr.data.build_ranking_dataset(
        file_pattern=path,
        data_format=tfr.data.ELWC,
        batch_size=_BATCH_SIZE,
        list_size=_LIST_SIZE,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=False,
        num_epochs=num_epochs)
  features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
  label = tf.cast(label, tf.float32)

  return features, label

input_fn('Data/Raw/tfrecords/train.tfrecords')
  
####
tf.train.Example


