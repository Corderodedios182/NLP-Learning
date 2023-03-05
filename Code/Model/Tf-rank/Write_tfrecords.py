# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:51:16 2023

@author: corde
"""

#####
import os
import pandas as pd
import pathlib
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_text as tf_text
from tensorflow_serving.apis import input_pb2

os.chdir(r"C:\Users\corde\Documents\0_Github\NLP-Learning")

datos = pd.read_csv('Data/Raw/tfrecords/datos_a_corregir.csv').loc[:,['QID','TEXTO_COMPARACION', 'D_EVENTO','RANK']]
datos['RANK'] = datos['RANK'].astype('int')
datos.columns = ['qid','query_tokens','document_tokens','relevance']

#example = datos[datos["qid"] == 1]

types = {'query_tokens': 'str',
         'document_tokens':'str',
         'relevance':'int'}

values_list = list(types.values())
keys_list = list(types.keys())

datos_values = datos.values

##############################
# -- Definir preprocessor -- #
##############################

class LookUpTablePreprocessor(tfr.keras.model.Preprocessor):

  def __init__(self, vocab_file, vocab_size, embedding_dim):
    self._vocab_file = vocab_file
    self._vocab_size = vocab_size
    self._embedding_dim = embedding_dim

  def __call__(self, context_inputs, example_inputs, mask):
    list_size = tf.shape(mask)[1]
    lookup = tf.keras.layers.StringLookup(
        max_tokens=self._vocab_size,
        vocabulary=self._vocab_file,
        mask_token=None)
    embedding = tf.keras.layers.Embedding(
        input_dim=self._vocab_size,
        output_dim=self._embedding_dim,
        embeddings_initializer=None,
        embeddings_constraint=None)
    # StringLookup and Embedding are shared over context and example features.
    context_features = {
        key: tf.reduce_mean(embedding(lookup(value)), axis=-2)
        for key, value in context_inputs.items()
    }
    example_features = {
        key: tf.reduce_mean(embedding(lookup(value)), axis=-2)
        for key, value in example_inputs.items()
    }
    return context_features, example_features

_VOCAB_FILE = 'Data/Raw/tfrecords/vocab_spanish.txt' #-> el vocab tiene origen en los datos originales
_VOCAB_SIZE = len(pathlib.Path(_VOCAB_FILE).read_text().split())

preprocessor = LookUpTablePreprocessor(_VOCAB_FILE, _VOCAB_SIZE, 20)

tokenizer = tf_text.BertTokenizer(_VOCAB_FILE)
example_tokens = tokenizer.tokenize("Hojalata invadiendo senda peatonal posibilita ocurrencia de incidente/accidente.".lower())

print(example_tokens)
print(tokenizer.detokenize(example_tokens))

######################################################################

###############
# Encoder W2V #
###############

import os
os.chdir(r"Code\DataPrep\utils")
from settings import *

os.chdir(r"C:/Users/corde/Documents/0_Github/NLP-Learning/Code/DataPrep")
import importData as importData
import preprocessing as preprocessing
import exploratoryData as exploratoryData
import dataModeling as dataModeling

import os
from settings import *
from text_utils import *

def vectorizing_data(
    datos_a_procesar: pd.DataFrame,
    model_w2v: model_w2v,
    pdt: ProcesadorDeTexto(),
) -> pd.DataFrame:
    """Vectorización de documentos, consultas y operaciones de extracción de características.
       Parameters    ----------    
       datos_a_procesar: pd.DataFrame
       _description_    pdt: new_text_utils.ProcesadorDeTexto
       _description_
       model_w2v: gensim.models.word2vec.Word2Vec
       _description_    
       Returns    -------
       pd.DataFrame        _description_    """    
    
    datos_a_procesar['query_tokens_encoder'] = \
        datos_a_procesar['query_tokens'].apply(
            func=pdt.vectorize,
            model_w2v=model_w2v,
        )
        
    datos_a_procesar['document_tokens_encoder'] = \
        datos_a_procesar['query_tokens'].apply(
            func=pdt.vectorize,
            model_w2v=model_w2v,
        )
        
    return datos_a_procesar

datos = vectorizing_data(datos_a_procesar = datos,
                         model_w2v = model_w2v,
                         pdt = ProcesadorDeTexto()).reset_index()

######################################################################        

#########################
#Escritura en Tf-records#
#
# No necesariamente tiene que poner sus datos en el formato de cadena mencionado
# anteriormente y es posible crear un objeto ELWC mediante programación al proporcionar
# una instancia tf.train.Example como su ejemplo de contexto y una lista de
# tf.train.Example para las características de sus documentos.
# Defina algunas funciones para envolver sus valores sin procesar dentro del tipo
# tf.train.Feature con los tipos respectivos:

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

# Luego, puede usar tf.io.TFRecordWriter y escribir tantos ELWC
# como desee (es decir, un ELWC por consulta) en el mismo registro TF, p.

#Escribir todos los datos

with tf.io.TFRecordWriter("C:/Users/corde/Documents/0_Github/NLP-Learning/Data/Raw/tfrecords/input_tfranking.tfrecords") as writer:
    
    context_dict = {}
    example_dict = {}
    
    ELWC = input_pb2.ExampleListWithContext()
    example_features = ELWC.examples.add()
    #example_context = ELWC.context
    
    #k = 3
    #
    for k in datos["qid"].unique():
        
        example = datos[datos["qid"] == k].reset_index().drop("index", axis = 1)
        
        for j in list(range(example.shape[0])):
            
            ELWC = input_pb2.ExampleListWithContext()
            example_features = ELWC.examples.add()
            
            example_dict["document"] = _bytes_feature([example.loc[j,"document_tokens"].encode('utf-8')])
            exampe_proto = tf.train.Example(features=tf.train.Features(feature = example_dict))
            
            example_dict["document_bert_encoder_outputs"] = _float_feature( example.loc[j,"query_tokens_encoder"] )
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
            
            example_dict["document_id"] = _bytes_feature([str(example.loc[j,"qid"]).encode('utf-8')])
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
            
            #Posición en el diccionario vocab.
            document_input = list(tokenizer.tokenize(str(example.loc[j,"document_tokens"]).encode('utf-8').lower()).to_list())[0]
            document_input_ids = []
            [document_input_ids.append(x[0]) for x in document_input]
            example_dict["document_input_ids"] = _int64_feature( document_input_ids )
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
            
            example_dict["document_input_mask"] = _int64_feature( [1]*len(document_input_ids) )
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
            
            example_dict["document_segment_ids"] = _int64_feature( [0]*len(document_input_ids) )
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
            
            example_dict["document_tokens"] = _bytes_feature( str(example.loc[j,"document_tokens"]).encode('utf-8').lower().split() )
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
                    
            example_dict["relevance"] = _int64_feature([example.loc[j,"relevance"]])
            exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
            
            example_features.CopyFrom(exampe_proto)
            
            writer.write(ELWC.SerializeToString())
            print("------- Escritura examples -----------", exampe_proto)
            
            if j == max(list(range(example.shape[0]))):
                
                ELWC = input_pb2.ExampleListWithContext()
                
                context_dict["query"] = _bytes_feature([ example.loc[0, "query_tokens"].encode('utf-8') ])
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
                
                context_dict["query_bert_encoder_outputs"] = _float_feature( example.loc[0, "document_tokens_encoder"] )
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
                
                context_dict["query_id"] = _int64_feature([example.loc[0, "qid"]])
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
                
                query_input = list(tokenizer.tokenize(str(example.loc[j,"query_tokens"]).encode('utf-8').lower()).to_list())[0]
                query_input_ids = []
                [query_input_ids.append(x[0]) for x in query_input]
                
                context_dict["query_input_ids"] = _int64_feature( query_input_ids )
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
                
                context_dict["query_input_mask"] = _int64_feature( [1] * len(query_input_ids) )
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
               
                context_dict["query_segment_ids"] = _int64_feature( [0] * len(query_input_ids) )
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
                
                context_dict["query_tokens"] = _bytes_feature( str(example.loc[j,"query_tokens"]).encode('utf-8').lower().split() )
                context_proto = tf.train.Example(features=tf.train.Features(feature = context_dict))
            
                print("######### Escritura context ##############", context_proto)
           
                ELWC.context.CopyFrom(context_proto)
            
                writer.write(ELWC.SerializeToString())
            
tmp = datos[datos["document_tokens"].str.contains("barras")]
            
####
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

read_and_print_tf_record("input_tfranking.tfrecords", 5)








