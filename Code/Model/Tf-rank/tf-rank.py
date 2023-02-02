# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 22:21:29 2023

@author: cflorelu
"""

import pandas as pd
import os

datos =  pd.read_parquet("C:/Users/corde/OneDrive/Documents/Model-20230131T172415Z-001/Model/Data/Raw/búsquedas_con_retro_recortado.parquet")
datos["KEY"] = datos["AM"] + "_" + datos["C_EVENTO"].astype(str)

datos.head()

#Preprocesamiento

import os
os.chdir(r"Code\DataPrep\utils")
from settings import *

os.chdir(r"C:\Users\cflorelu\Documents\1_Ternium\Resumenes-Python\04_Machine Learning Scientist\12_Natural_Lenguague\Model\Code\DataPrep")
import importData as importData
import preprocessing as preprocessing
import exploratoryData as exploratoryData
import dataModeling as dataModeling
#from trainModel import *
#from reporting import *

os.chdir(r"C:\Users\cflorelu\Documents\1_Ternium\Resumenes-Python\04_Machine Learning Scientist\12_Natural_Lenguague\Model")

#-- preprocessing --#

#datos = preprocessing.reordering_data(datos_a_corregir= datos,
#                                      evaluaciones_sin_ceros = False) #Si es True (descarta los ceros)

datos_a_corregir = datos
evaluaciones_sin_ceros = True

if evaluaciones_sin_ceros == True:
    datos_a_corregir = datos_a_corregir[datos_a_corregir["EVALUACION"] != 0]
    
datos_a_corregir["C_EVENTO_UNIQUE"] = datos_a_corregir["TEXTO_COMPARACION"] + " - " + datos_a_corregir["D_EVENTO"]
    
tmp = datos_a_corregir["TEXTO_COMPARACION"] + " - " + datos_a_corregir["D_EVENTO"]
tmp = pd.DataFrame(tmp.drop_duplicates())
tmp["C_EVENTO_NEW"] = list(range(tmp.shape[0]))
tmp.columns = ["C_EVENTO_UNIQUE","C_EVENTO_NEW"]
    
datos_a_corregir = datos_a_corregir.merge(tmp,
                                          on = 'C_EVENTO_UNIQUE')
    
datos_a_corregir["AM-C_EVENTO_UNIQUE"] = datos_a_corregir["AM"].astype(str) + "-" + datos_a_corregir["C_EVENTO_UNIQUE"].astype(str)
datos_a_corregir['AM-TEXTO_COMPARACION'] = datos_a_corregir["AM"].astype(str) + "-" + datos_a_corregir["TEXTO_COMPARACION"].astype(str)
    
# Ordenamiento de datos para facilitar el uso de métodos con grupos
datos_a_corregir = datos_a_corregir.sort_values(by = ['C_EVENTO_NEW','AM-C_EVENTO_UNIQUE', 'F_CREATE'])
    
# Diccionarios utilitarios, ayudarán más adelante a dar formato a ciertos datos
textos_comparacion_únicos = \
    datos_a_corregir['AM-TEXTO_COMPARACION'].unique()
        
textos_comparacion_únicos = \
    textos_comparacion_únicos.tolist()
        
textos_comparacion_idx_dicc = \
    {
        elem: idx
        for (idx, elem)
        in enumerate(
            iterable=textos_comparacion_únicos,
            start=1,
            )
        }

# Creación de la columna QID (de identificación), utiliza valores de la columna
# TEXTO_COMPARACION reemplazando el texto de consulta por un número
datos_a_corregir['QID'] = \
    datos_a_corregir[['AM-TEXTO_COMPARACION']].replace(
            to_replace=textos_comparacion_idx_dicc,
        )
        
# Reordenamiento de columnas
datos_a_corregir = \
    datos_a_corregir[
        [
            'C_EVENTO',
            'C_EVENTO_NEW',
            'C_EVENTO_UNIQUE',
            'AM',
            'QID',
            'TEXTO_COMPARACION',
            'D_EVENTO',
            'SIMILITUD',
            'EVALUACION',
            #'gaussian'
            ]
        ]

# Aquí se suman todos los C_EVENTO correspondiente a su grupo QID
datos_a_corregir['EVALUACION_SUM'] = \
    datos_a_corregir.groupby(
        by=['AM', 'C_EVENTO_UNIQUE']
        )['EVALUACION'].transform('sum')

# Se borran los que están duplicados, dado que ya se sumaron todas las 
#  instancias
datos_a_corregir = datos_a_corregir.drop(["EVALUACION"], axis = 1)
    
datos_a_corregir = \
    datos_a_corregir.drop_duplicates(
        subset = \
            [
                    'C_EVENTO_NEW',
                    'C_EVENTO_UNIQUE',
                    'AM',
                    'QID',
                    'TEXTO_COMPARACION',
                    'D_EVENTO',
                ],
            keep='first',
        )
    
datos_a_corregir['RANK'] = datos_a_corregir.groupby("QID")["EVALUACION_SUM"].rank('first', ascending =  False)
    
datos_a_corregir = datos_a_corregir.sort_values(["QID","RANK"])

datos_a_corregir = preprocessing.vectorizing_data(datos_a_procesar = datos_a_corregir,
                                                  model_w2v = model_w2v,
                                                  pdt = ProcesadorDeTexto()).reset_index()

datos_a_corregir.head()

## Nueva característica a los datos.

#1. Variable gaussian : Generación de un cluster para la variable Texto_Comparación, usando el modelo gmm (https://prodlsandbox.blob.core.windows.net/safety-update-data-prod/text_updater/2022-12-09/)
    
#2. Distancia euclidian y manhattan sobre vectores de texto_comparación y eventos.

#3. Top de argsort sobre la diff de vectores.

from sklearn.mixture import GaussianMixture
import numpy as np

_GMM_FILE = "new_gmm"

DATA_FOLDER = 'Code/DataPrep/utils/models'

MODEL_GMM_MEANS = \
    os.path.join(DATA_FOLDER, 'gmm', "{}_means.npy".format(_GMM_FILE))
MODEL_GMM_COVARS = \
    os.path.join(DATA_FOLDER, 'gmm', "{}_covariances.npy".format(_GMM_FILE))
MODEL_GMM_WEIGHTS = \
    os.path.join(DATA_FOLDER, 'gmm', "{}_weights.npy".format(_GMM_FILE))
    
## Carga de GMM
print("Cargando Cluster model")

means = np.load(MODEL_GMM_MEANS)
covars = np.load(MODEL_GMM_COVARS)
weights = np.load(MODEL_GMM_WEIGHTS)   

loaded_gmm = \
        GaussianMixture(
            n_components=len(means),
            covariance_type='full'
        )

loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covars))
loaded_gmm.covariances_ = covars
loaded_gmm.means_ = means
loaded_gmm.weights_ = weights
means = []; covars = []; weights = []

#prediccion de los dos clusters mas probables
query_texto_comparacion = list(datos_a_corregir["TEXTO_COMPARACION_VECT"])

prediction_texto_comparacion = loaded_gmm.predict_proba(np.asarray(query_texto_comparacion))

gaussian_1 = []
gaussian_2 = []
gaussian_3 = []
gaussian_4 = []
gaussian_5 = []
gaussian_6 = []
gaussian_7 = []
gaussian_8 = []
gaussian_9 = []
gaussian_10 = []
gaussian_11 = []

for i in range(0,len(prediction_texto_comparacion)):

     gaussian_1.append(prediction_texto_comparacion[i].argsort()[-1])
     gaussian_2.append(prediction_texto_comparacion[i].argsort()[-2])
     gaussian_3.append(prediction_texto_comparacion[i].argsort()[-3])
     gaussian_4.append(prediction_texto_comparacion[i].argsort()[-4])
     gaussian_5.append(prediction_texto_comparacion[i].argsort()[-5])
     gaussian_6.append(prediction_texto_comparacion[i].argsort()[-6])
     gaussian_7.append(prediction_texto_comparacion[i].argsort()[-7])
     gaussian_8.append(prediction_texto_comparacion[i].argsort()[-8])
     gaussian_9.append(prediction_texto_comparacion[i].argsort()[-9])
     gaussian_10.append(prediction_texto_comparacion[i].argsort()[-10])
     gaussian_11.append(prediction_texto_comparacion[i].argsort()[-11])

datos_a_corregir["gaussian_1_texto"] = gaussian_1
datos_a_corregir["gaussian_2_texto"] = gaussian_2
datos_a_corregir["gaussian_3_texto"] = gaussian_3
datos_a_corregir["gaussian_4_texto"] = gaussian_4
datos_a_corregir["gaussian_5_texto"] = gaussian_5
datos_a_corregir["gaussian_6_texto"] = gaussian_6
datos_a_corregir["gaussian_7_texto"] = gaussian_7
datos_a_corregir["gaussian_8_texto"] = gaussian_8
datos_a_corregir["gaussian_9_texto"] = gaussian_9
datos_a_corregir["gaussian_10_texto"] = gaussian_10
datos_a_corregir["gaussian_11_texto"] = gaussian_11

datos_a_corregir.head()

query_evento = list(datos_a_corregir["D_EVENTO_VECT"])

prediction_evento = loaded_gmm.predict_proba(np.asarray(query_evento))

gaussian_1 = []
gaussian_2 = []
gaussian_3 = []
gaussian_4 = []
gaussian_5 = []
gaussian_6 = []
gaussian_7 = []
gaussian_8 = []
gaussian_9 = []
gaussian_10 = []
gaussian_11 = []

for i in range(0,len(prediction_evento)):

     gaussian_1.append(prediction_evento[i].argsort()[-1])
     gaussian_2.append(prediction_evento[i].argsort()[-2])
     gaussian_3.append(prediction_evento[i].argsort()[-3])
     gaussian_4.append(prediction_evento[i].argsort()[-4])
     gaussian_5.append(prediction_evento[i].argsort()[-5])
     gaussian_6.append(prediction_evento[i].argsort()[-6])
     gaussian_7.append(prediction_evento[i].argsort()[-7])
     gaussian_8.append(prediction_evento[i].argsort()[-8])
     gaussian_9.append(prediction_evento[i].argsort()[-9])
     gaussian_10.append(prediction_evento[i].argsort()[-10])
     gaussian_11.append(prediction_evento[i].argsort()[-10])

datos_a_corregir["gaussian_1_evento"] = gaussian_1
datos_a_corregir["gaussian_2_evento"] = gaussian_2
datos_a_corregir["gaussian_3_evento"] = gaussian_3
datos_a_corregir["gaussian_4_evento"] = gaussian_4
datos_a_corregir["gaussian_5_evento"] = gaussian_5
datos_a_corregir["gaussian_6_evento"] = gaussian_6
datos_a_corregir["gaussian_7_evento"] = gaussian_7
datos_a_corregir["gaussian_8_evento"] = gaussian_8
datos_a_corregir["gaussian_9_evento"] = gaussian_9
datos_a_corregir["gaussian_10_evento"] = gaussian_10
datos_a_corregir["gaussian_11_evento"] = gaussian_11

#Distancia euclidian y manhattan sobre vectores de texto_comparación y eventos.
datos_a_corregir["dist_euclidian"] = datos_a_corregir.apply(lambda x: np.linalg.norm(np.array(x["TEXTO_COMPARACION_VECT"]) - np.array(x["D_EVENTO_VECT"])), axis = 1)

#Top de argsort sobre la diff de vectores.
argsort_1 = []
argsort_2 = []
argsort_3 = []
argsort_4 = []
argsort_5 = []

for i in range(0,datos_a_corregir.shape[0]):

    argsort_1.append(datos_a_corregir["DIFF_TEXTO_COMPARACION_VECT_&_D_EVENTO_VECT"].apply(lambda x : np.array(x).argsort())[i][-1])
    argsort_2.append(datos_a_corregir["DIFF_TEXTO_COMPARACION_VECT_&_D_EVENTO_VECT"].apply(lambda x : np.array(x).argsort())[i][-2])
    argsort_3.append(datos_a_corregir["DIFF_TEXTO_COMPARACION_VECT_&_D_EVENTO_VECT"].apply(lambda x : np.array(x).argsort())[i][-3])
    argsort_4.append(datos_a_corregir["DIFF_TEXTO_COMPARACION_VECT_&_D_EVENTO_VECT"].apply(lambda x : np.array(x).argsort())[i][-4])
    argsort_5.append(datos_a_corregir["DIFF_TEXTO_COMPARACION_VECT_&_D_EVENTO_VECT"].apply(lambda x : np.array(x).argsort())[i][-5])

datos_a_corregir["argsort_max1"] = argsort_1
datos_a_corregir["argsort_max2"] = argsort_2
datos_a_corregir["argsort_max3"] = argsort_3
datos_a_corregir["argsort_max4"] = argsort_4
datos_a_corregir["argsort_max5"] = argsort_5

datos_a_corregir.head()
datos_a_corregir.columns

## Preparación de datos para el Modelo.

#-- dataModeling --#
x_entrna, y_entrna, x_valida, y_valida, x_prueba, y_prueba = dataModeling.split_data(datos_a_dividir = datos_a_corregir, 
                                                                                     p_explr_prueba = .8,
                                                                                     p_entrn_valida = .8,
                                                                                     )

noms_cols_caracts = ['gaussian_1_texto',
                     'gaussian_2_texto',
                     'gaussian_3_texto',
                     'gaussian_4_texto',
                     'gaussian_5_texto',
                     'gaussian_6_texto',
                     'gaussian_7_texto',
                     'gaussian_8_texto',
                     'gaussian_9_texto',
                     'gaussian_10_texto',
                     'gaussian_11_texto',
                     'gaussian_1_evento',
                     'gaussian_2_evento',
                     'gaussian_3_evento',
                     'gaussian_4_evento',
                     'gaussian_5_evento',
                     'gaussian_6_evento',
                     'gaussian_7_evento',
                     'gaussian_8_evento',
                     'gaussian_9_evento',
                     'gaussian_10_evento',
                     'gaussian_11_evento',
                     'TEXTO_COMPARACION_VECT',
                     'D_EVENTO_VECT',
                     'DIFF_TEXTO_COMPARACION_VECT_&_D_EVENTO_VECT',
                     'SIMILITUD',
                     'dist_euclidian',
                     'argsort_max1',
                     'argsort_max2',
                     'argsort_max3',
                     'argsort_max4',
                     'argsort_max5']

# Listas con las características, etiquetas y grupos utilizadas en el entrenamiento del modelo
feats_entrna, labels_entrna, qids_entrna = dataModeling.obtener_CEQs(X = x_entrna,
                                                                     y = y_entrna,
                                                                     noms_cols_caracts = noms_cols_caracts)

feats_valida, labels_valida, qids_valida = dataModeling.obtener_CEQs(X = x_valida,
                                                                     y = y_valida,
                                                                     noms_cols_caracts = noms_cols_caracts)

feats_prueba, labels_prueba, qids_prueba = dataModeling.obtener_CEQs(X = x_prueba,
                                                                     y = y_prueba,
                                                                     noms_cols_caracts = noms_cols_caracts)

x_entrna.head()

pd.DataFrame(feats_entrna)

#--Validación sobre un mismo dataset para los experimentos--#
#x_entrna.groupby(["QID","gaussian"]).count()
#x_entrna[x_entrna["QID"] == 3]

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax1 = plt.subplots(figsize=(20,5))

tmp = x_entrna.groupby(["QID"], as_index = False).agg({"EVALUACION_SUM":"sum"}).sort_values(["EVALUACION_SUM"])

plot_order = tmp["QID"].unique()

sns.barplot(data = tmp,
            x = 'QID',
            y = 'EVALUACION_SUM',
            order = plot_order).set(title='QID x_entren')

plt.show()

#-- modelo tf-rank --#

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Arquitectura del modelo

model = Sequential() #Instanciando el modelo

model.add(Dense(300, input_shape = (329,), activation = 'relu')) #Entradas del modelo

model.add(Dense(300, activation = 'relu')) #Primera capa

model.add(Dense(1)) #Output del modelo

#2. Compilado del modelo (optimizador y función de perdida)

model.compile(optimizer = 'adam', loss = 'mse')

#3. Entrenamiento

model.fit(feats_entrna, labels_entrna, epochs = 15)

#4. Evaluación del modelo

model.evaluate(feats_prueba, labels_prueba)

#5. Predicción
model.predict(feats_prueba)

#-- Evaluación NDCG's --#
x_entrna = x_entrna.reset_index()
x_entrna["RANK"] = pd.DataFrame(labels_entrna).reset_index().iloc[:,1]
x_entrna["y_pred"] = 0
x_entrna["y_pred"] = pd.DataFrame(model.predict(feats_entrna)).astype(float).reset_index().iloc[:,1]
x_entrna["rank_tf"] = (x_entrna.groupby('QID')['y_pred'].rank(method='dense', ascending = True).astype(int))
x_entrna["datos"] = 'entrenamiento'

x_entrna[x_entrna["QID"] == 2].loc[:,["QID","TEXTO_COMPARACION","D_EVENTO","EVALUACION_SUM","RANK","rank_tf"]]

#Valida
x_valida = x_valida.reset_index()
x_valida["RANK"] = pd.DataFrame(labels_valida).reset_index().iloc[:,1]
x_valida["y_pred"] = 0
x_valida["y_pred"] = pd.DataFrame(model.predict(feats_valida)).astype(float).reset_index().iloc[:,1]
x_valida["rank_tf"] = (x_valida.groupby('QID')['y_pred'].rank(method='dense', ascending = True).astype(int))
x_valida["datos"] = 'valida'

#Prueba
x_prueba = x_prueba.reset_index()
x_prueba["RANK"] = pd.DataFrame(labels_prueba).reset_index().iloc[:,1]
x_prueba["y_pred"] = 0
x_prueba["y_pred"] = pd.DataFrame(model.predict(feats_prueba)).astype(float).reset_index().iloc[:,1]
x_prueba["rank_tf"] = (x_prueba.groupby('QID')['y_pred'].rank(method='dense', ascending = True).astype(int))
x_prueba["datos"] = 'prueba'

#Union
datos_l2r = pd.concat([x_entrna, x_valida, x_prueba])

datos_l2r['rank_w2v'] = datos_l2r.groupby("QID")["SIMILITUD"].rank('first', ascending =  False)
datos_l2r['rank_w2v'] = datos_l2r['rank_w2v'].astype(int)
datos_l2r['RANK'] = datos_l2r['RANK'].astype(int)

datos_l2r.head()
datos_l2r["datos"].value_counts()

#ndcg por grupo QID

qid_count = datos_l2r.groupby(["QID"], as_index = False).agg({"D_EVENTO" : "count",
                                                              "EVALUACION_SUM" : "sum"})

qid = list(set(datos_l2r["QID"]))

ndcg_tf = []
ndcg_w2v = []

qids = []
dato = []

i = 1

for i in range(len(qid)):
    
    dat = datos_l2r[datos_l2r["QID"] == qid[i]]['datos'].unique()[0]
    rank_origin = np.asarray([list(datos_l2r[datos_l2r["QID"] == qid[i]].loc[:,['RANK']]['RANK'])])
    rank_tf = np.asarray([list(datos_l2r[datos_l2r["QID"] == qid[i]].loc[:,['rank_tf']]['rank_tf'])])
    rank_w2v = np.asarray([list(datos_l2r[datos_l2r["QID"] == qid[i]].loc[:,['rank_w2v']]['rank_w2v'])])
    
    dato.append(dat)
    qids.append(qid[i])
    
    if rank_origin.shape[1] == 1:
        
        ndcg_tf.append(1)
        ndcg_w2v.append(1)

    else:
        
        ndcg_tf.append(ndcg_score(rank_tf, rank_origin))
        ndcg_w2v.append(ndcg_score(rank_w2v, rank_origin))

df_ndcg = pd.DataFrame(zip(qids, ndcg_tf, ndcg_w2v, dato),
                       columns = ['qids','ndcg_tf', 'ndcg_w2v','dato'])

df_ndcg['dif_ndcg'] = df_ndcg['ndcg_tf'] - df_ndcg['ndcg_w2v']

df_ndcg = df_ndcg.merge(qid_count,
                        left_on = "qids",
                        right_on = "QID")

#df_ndcg = df_ndcg[~df_ndcg["qids"].isin([70,33,36])]
df_ndcg[df_ndcg["qids"] == 123]


#Resultados 

print("tf-rank modelo simple")
print("")
tmp = df_ndcg.groupby(["dato"], as_index = False).agg({'ndcg_tf':'mean'})
tmp["ndcg_tf"] = round(tmp["ndcg_tf"] * 100,1)
print(tmp)

print("-------")

tmp = df_ndcg.groupby(["dato"], as_index = False).agg({'ndcg_w2v':'mean'})
tmp["ndcg_w2v"] = round(tmp["ndcg_w2v"] * 100,1)
print(tmp)

# -- Tubería de clasificación -- #

import pathlib
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_text as tf_text
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format

# -- Feature Specification -- #
# Funciones son abstracciones de TensorFlow que se utilizan para capturar información detallada sobre cada función.

#API pública para el espacio de nombres tf.io
context_feature_spec = {
    "query_tokens": tf.io.RaggedFeature(dtype=tf.string), #Configuración para pasar una función de entrada RaggedTensor.
}

example_feature_spec = {
    "document_tokens":
        tf.io.RaggedFeature(dtype=tf.string), #Configuración para pasar una función de entrada RaggedTensor.
}
    
label_spec = (
    "relevance",
    tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=-1) #Configuración para analizar una característica de entrada de longitud fija.
)

# -- Input creator -- #
#InputCreator con especificaciones de características.

input_creator = tfr.keras.model.FeatureSpecInputCreator(
    context_feature_spec, example_feature_spec)

input_creator()

#  -- Preprocessor -- #
#Interfaz para preprocesamiento de características.
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

_VOCAB_FILE = 'C:/Users/corde/OneDrive/Documents/Model-20230131T172415Z-001/Model/Data/Raw/tmp/vocab.txt'
_VOCAB_SIZE = len(pathlib.Path(_VOCAB_FILE).read_text(encoding=('utf8')).split())

preprocessor = LookUpTablePreprocessor(_VOCAB_FILE, _VOCAB_SIZE, 20)

tokenizer = tf_text.BertTokenizer(_VOCAB_FILE)
example_tokens = tokenizer.tokenize("Hello TensorFlow!".lower())

print(example_tokens)
print(tokenizer.detokenize(example_tokens))

# -- Score -- #
#Anotador univariado usando DNN
scorer = tfr.keras.model.DNNScorer(
    hidden_layer_dims=[64, 32, 16],
    output_units=1,
    activation=tf.nn.relu,
    use_batch_norm=True)


# -- Model builder -- #
#Construye un tf.keras.Model.
#Hereda de: ModelBuilderWithMask, AbstractModelBuilder

model_builder = tfr.keras.model.ModelBuilder(
    input_creator=input_creator,
    preprocessor=preprocessor,
    scorer=scorer,
    mask_feature_name="example_list_mask",
    name="antique_model",
)

model = model_builder.build()

tf.keras.utils.plot_model(model, expand_nested=True)

# -- De csv a tfrecords -- #

# -- De tfrecords a dataframe -- #

# -- Hiperparámetros de datos -- #

dataset_hparams = tfr.keras.pipeline.DatasetHparams(
    train_input_pattern="/tmp/train.tfrecords", #TFRecord es un formato simple para almacenar una secuencia de registros binarios.
    valid_input_pattern="/tmp/test.tfrecords", 
    train_batch_size=32,
    valid_batch_size=32,
    list_size=50,
    dataset_reader=tf.data.TFRecordDataset)

# -- Dataset_builder -- #
#Generar conjuntos de datos ELWC usando feature_spec

dataset_builder = tfr.keras.pipeline.SimpleDatasetBuilder(
    context_feature_spec,
    example_feature_spec,
    mask_feature_name="example_list_mask",
    label_spec=label_spec,
    hparams=dataset_hparams)

ds_train = dataset_builder.build_train_dataset()
ds_train.element_spec

%load_ext tensorboard
%tensorboard --logdir="/tmp/ranking_model_dir" --port 12345







