# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:51:16 2023

@author: corde
"""
import os
import pandas as pd
import tensorflow as tf

os.chdir(r"C:\Users\corde\Documents\0_Github\NLP-Learning")

datos = pd.read_csv('Data/Raw/tfrecords/datos_a_corregir.csv').loc[:,['QID','TEXTO_COMPARACION', 'D_EVENTO','RANK']]
datos['RANK'] = datos['RANK'].astype('int')
datos.columns = ['qid','query_tokens','document_tokens','relevance']

example = datos[datos["query_tokens"] == 'reunión seguridad']

types = {'query_tokens': 'str',
         'document_tokens':'str',
         'relevance':'int'}

values_list = list(types.values())
keys_list = list(types.keys())

datos_values = datos.values

#No ocupo : elwc = input_pb2.ExampleListWithContext()
#2 tr.train.Example  una para query y otra para documentos 

#Ejemplo para n = 1 :
    
YOUR_QUERY_TOKENS = example["query_tokens"].apply(lambda x : x.encode('utf-8'))#[0]
YOUR_DOCUMENT_TOKENS = example["document_tokens"].apply(lambda x : x.encode('utf-8')) 
YOUR_RELEVANCE_TOKENS = example["relevance"]

#No necesariamente tiene que poner sus datos en el formato de cadena mencionado
#anteriormente y es posible crear un objeto ELWC mediante programación al proporcionar
#una instancia tf.train.Example como su ejemplo de contexto y una lista de
#tf.train.Example para las características de sus documentos.

#Defina algunas funciones para envolver sus valores sin procesar dentro del tipo
#tf.train.Feature con los tipos respectivos:

def _bytes_feature(value_list):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value_list, type(tf.constant(0))):
        value_list = value_list.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))

def _float_feature(value_list):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

def _int64_feature(value_list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

#Luego, puede usar tf.io.TFRecordWriter y escribir tantos ELWC
#como desee (es decir, un ELWC por consulta) en el mismo registro TF, p.

#(Tenga en cuenta que este es un código muy aproximado, pero debería darle una idea de
#cómo usar input_pb2.ExampleListWithContext):
    
from tensorflow_serving.apis import input_pb2

YOUR_QUERY_TOKENS = []
YOUR_DOCUMENT_TOKENS = []
YOUR_RELEVANCE_TOKENS = []

with tf.io.TFRecordWriter("input_tfranking.tfrecords") as writer:
    
    context_dict = {}
    context_dict["query_toknes"] = _bytes_feature(YOUR_QUERY_TOKENS)
    context_proto = tf.train.Example(features=tf.train.Features(feature=context_dict))
         
    ELWC = input_pb2.ExampleListWithContext()
    ELWC.context.CopyFrom(context_proto)
            
    example_features = ELWC.examples.add()

    example_dict = {}
    example_dict["document_tokens"] = _bytes_feature(YOUR_DOCUMENT_TOKENS)
    exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
    example_features.CopyFrom(exampe_proto)
        
    relevance_dict = {}
    relevance_dict["relevance"] = _int64_feature(YOUR_RELEVANCE_TOKENS)
    relevance_proto = tf.train.Example(features=tf.train.Features(feature=relevance_dict))
    example_features.CopyFrom(relevance_proto)
    
    writer.write(ELWC.SerializeToString())    

#Otra forma es aprovechar el constructor ExampleListWithContext y simplemente
#pasarle el contexto tf.train.Example 
#y una lista de tf.train.Example para los documentos,
#que sospecho que es más eficiente que pasar por la operación CopyFrom:
  
#ELWC = input_pb2.ExampleListWithContext(context=context_proto,
#                                        examples=example_list)

#Después de escribir el registro TF, puede verificar su cordura leyéndolo e
#imprimiendo algunos ELWC según sea necesario, por ejemplo:
    
def read_and_print_tf_record(target_filename, num_of_examples_to_read):
    filenames = [target_filename]
    tf_record_dataset = tf.data.TFRecordDataset(filenames)
    
    for raw_record in tf_record_dataset.take(num_of_examples_to_read):
        example_list_with_context = input_pb2.ExampleListWithContext()
        example_list_with_context.ParseFromString(raw_record.numpy())
        print(example_list_with_context)
        
read_and_print_tf_record("input_tfranking.tfrecords", 1)

input_tfranking = tf_record_dataset = tf.data.TFRecordDataset("input_tfranking.tfrecords")

import tensorflow_ranking as tfr

def parse_elwc(elwc):
  return tfr.data.parse_from_example_list(
      [elwc],
      list_size=2,
      context_feature_spec={"query_tokens": tf.io.RaggedFeature(dtype=tf.string)},
      example_feature_spec={
          "document_tokens":
              tf.io.RaggedFeature(dtype=tf.string),
          "relevance":
              tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
      },
      size_feature_name="_list_size_",
      mask_feature_name="_mask_")

parse_elwc(input_tfranking)




