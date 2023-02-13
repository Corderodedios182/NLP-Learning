# -*- coding: utf-8 -*-

import os
import pandas as pd
import tensorflow as tf

#tf.train.Example
#Es un proto estándar que almacena datos para entrenamiento e inferencia.Example    
#Fundamentalmente, un tf.train.Example es un mapeo {"string": tf.train.Feature} .

#Datos que tengo

os.chdir(r"C:\Users\corde\Documents\0_Github\NLP-Learning")

datos = pd.read_csv('Data/Raw/tfrecords/datos_a_corregir.csv').loc[:,['QID','TEXTO_COMPARACION', 'D_EVENTO','RANK']]
datos['RANK'] = datos['RANK'].astype('int')
datos.columns = ['qid','query_tokens','document_tokens','relevance']

types = {'qid':'int',
         'query_tokens': 'str',
         'document_tokens':'str',
         'relevance':'int'}

values_list = list(types.values())
keys_list = list(types.keys())

datos_values = datos.values

def create_tf_example(features, label):

    tf_example = tf.train.Example(features=tf.train.Features(feature={

        'query_tokens': tf.train.Feature(bytes_list = tf.train.BytesList(value = [features[1].encode('utf-8')])),
        'document_tokens': tf.train.Feature(bytes_list = tf.train.BytesList(value = [features[2].encode('utf-8')])),
        
        'relevane': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
                
        }))

    return tf_example

with tf.io.TFRecordWriter("dataset.tfrecords") as writer:

    for row in datos_values:
        
        query_tokens, document_tokens, relevance = row[1], row[2], row[3]
        
        example = create_tf_example(query_tokens, relevance)
     
        writer.write(example.SerializeToString())

    writer.close()

#Lectora del Tfrecords

raw_dataset = tf.data.TFRecordDataset('dataset.tfrecords')

for raw_record in raw_dataset.take(1):
  print(repr(raw_record))



    



