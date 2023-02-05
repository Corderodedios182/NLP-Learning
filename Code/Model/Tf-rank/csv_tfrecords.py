# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:25:39 2023

@author: corde
"""

import tensorflow as tf
import pandas as pd

cars_dataframe = pd.read_csv('Data/Raw/tfrecords/cars.csv', sep = ';')

cars_dataframe = cars_dataframe.loc[:,['Car']]

types = {'Car':'str'}
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

with tf.io.TFRecordWriter(path="cars.tfrecords") as writer:
  for _, row in cars_dataframe.iterrows():
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

cars_dataset = tf.data.TFRecordDataset("cars.tfrecords")
parsed_tfrecords = cars_dataset.map(map_func=parse_example)
for tfrecord in parsed_tfrecords:
  print(tfrecord['Car'].numpy())
  
#######



  
  
  
  
  
  