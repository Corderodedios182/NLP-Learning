# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:09:28 2023

@author: corde
"""

import pathlib

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_text as tf_text
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format

#wget -O "Data/Raw/tfrecords/train.tfrecords" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/ELWC/train.tfrecords"
#wget -O "Data/Raw/tfrecords/test.tfrecords" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking//ELWC/test.tfrecords"
#wget -O "Data/Raw/tfrecords/vocab.txt" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/vocab.txt"

# -- Especificar cacterísticas -- #

#Las especificaciones de características son abstracciones de TensorFlow que se utilizan para capturar información enriquecida sobre cada característica.

#Cree especificaciones de características para entidades de contexto, entidades de ejemplo y etiquetas, coherentes con los formatos de entrada para la clasificación, como el formato ELWC.

context_feature_spec = {
    "query_tokens": tf.io.RaggedFeature(dtype=tf.string),
}
example_feature_spec = {
    "document_tokens":
        tf.io.RaggedFeature(dtype=tf.string),
}
label_spec = (
    "relevance",
    tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=-1)
)


# -- Definir input_creator -- #

#input_creator crear diccionarios de contexto y ejemplo tf.keras.Inputs para las características de entrada definidas en y .context_feature_specexample_feature_spec

input_creator = tfr.keras.model.FeatureSpecInputCreator(
    context_feature_spec, example_feature_spec)

# -- Definir preprocessor -- #

#En el , los tokens de entrada se convierten en un vector activo a través de la capa de preprocesamiento Búsqueda de cadenas y, a continuación,
#se incrustan como un vector de incrustación a través de la capa de preprocesamiento Incrustación.
#Finalmente, calcule un vector de incrustación para la oración completa por el promedio de incrustaciones de tokens.preprocessor

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

_VOCAB_FILE = 'Data/Raw/tfrecords/vocab.txt'
_VOCAB_SIZE = len(pathlib.Path(_VOCAB_FILE).read_text(encoding = 'utf-8').split())

preprocessor = LookUpTablePreprocessor(_VOCAB_FILE, _VOCAB_SIZE, 20)

tokenizer = tf_text.BertTokenizer(_VOCAB_FILE)
example_tokens = tokenizer.tokenize("Hello TensorFlow!".lower())

print(example_tokens)
print(tokenizer.detokenize(example_tokens))

# -- Definir Scorer -- #
#En este ejemplo se utiliza un puntuador univariante de red neuronal profunda (DNN), predefinido en TensorFlow Ranking.

scorer = tfr.keras.model.DNNScorer(
    hidden_layer_dims=[64, 32, 16],
    output_units=1,
    activation=tf.nn.relu,
    use_batch_norm=True)

# -- Moodel Builder -- #

model_builder = tfr.keras.model.ModelBuilder(
    input_creator=input_creator,
    preprocessor=preprocessor,
    scorer=scorer,
    mask_feature_name="example_list_mask",
    name="antique_model",
)

model = model_builder.build()

tf.keras.utils.plot_model(model, expand_nested=True)

# -- Crear un generador de conjuntos de datos -- #

dataset_hparams = tfr.keras.pipeline.DatasetHparams(
    train_input_pattern="Data/Raw/tfrecords/train.tfrecords",
    valid_input_pattern="Data/Raw/tfrecords/test.tfrecords",
    train_batch_size=32,
    valid_batch_size=32,
    list_size=50,
    dataset_reader=tf.data.TFRecordDataset)

# -- Hacer dataset_builder : generar un conjunto e datos ELWC. -- #

dataset_builder = tfr.keras.pipeline.SimpleDatasetBuilder(
    context_feature_spec,
    example_feature_spec,
    mask_feature_name="example_list_mask",
    label_spec=label_spec,
    hparams=dataset_hparams)

ds_train = dataset_builder.build_train_dataset()
ds_train.element_spec

# -- Creando una canalización de clasificación -- #

#Especifique los hiperparámetros que se utilizarán para ejecutar la canalización creando un objeto.
#ranking_pipeline y pipeline_hparams

#Entrene el modelo con una velocidad de aprendizaje igual a 0.05 para 5 épocas con 1000 pasos en cada época usando.

#Evalúe el modelo en el conjunto de datos de validación durante 100 pasos después de cada época.
#Guarde el modelo entrenado en .approx_ndcg_loss MirroredStrategy ranking_model_dir

# -- PipelineHparams -- #

pipeline_hparams = tfr.keras.pipeline.PipelineHparams(
    model_dir="Data/Modeling/ranking_model_dir",
    num_epochs=5,
    steps_per_epoch=1000,
    validation_steps=100,
    learning_rate=0.05,
    loss="approx_ndcg_loss",
    strategy="MirroredStrategy")

# -- Ranking_pipeline -- #

#TensorFlow Ranking proporciona un modelo predefinido para apoyar el entrenamiento con estrategias pipeline.SimplePipeline

ranking_pipeline = tfr.keras.pipeline.SimplePipeline(
    model_builder,
    dataset_builder=dataset_builder,
    hparams=pipeline_hparams)

# -- Entrenar y evaluar el modelo -- #

#La función evalúa el modelo entrenado en el conjunto de datos de validación después de cada época.train_and_validate

ranking_pipeline.train_and_validate(verbose=1)

# -- Iniciando TensorBoard -- # 

%load_ext tensorboard
%tensorboard --logdir "Data/Modeling/ranking_model_dir" --port 8083
%reload_ext tensorboard

%tensorboard dev upload --logdir 'Data/Modeling/ranking_model_dir' --port 8083

# -- Generar predicciones y evaluar -- #

ds_test = dataset_builder.build_valid_dataset()

# Obtener características de entrada del primer lote de datos de prueba
for x, y in ds_test.take(1):
  break

#Cargue el modelo guardado y ejecute una predicción.
loaded_model = tf.keras.models.load_model("Data/Modeling/ranking_model_dir/export/latest_model")

# Predict ranking scores
scores = loaded_model.predict(x)
min_score = tf.reduce_min(scores)
scores = tf.where(tf.greater_equal(y, 0.), scores, min_score - 1e-5)

# Sort the answers by scores
sorted_answers = tfr.utils.sort_by_scores(
    scores,
    [tf.strings.reduce_join(x['document_tokens'], -1, separator=' ')])[0]

#Revisa las 5 respuestas principales para la pregunta número 4.
question = tf.strings.reduce_join(
    x['query_tokens'][4, :], -1, separator=' ').numpy()
top_answers = sorted_answers[4, :5].numpy()

print(
    f'Q: {question.decode()}\n' +
    '\n'.join([f'A{i+1}: {ans.decode()}' for i, ans in enumerate(top_answers)]))

