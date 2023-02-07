# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:49:00 2023

@author: corde
"""

import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_serving.apis import input_pb2

from google.protobuf import text_format

CONTEXT = text_format.Parse(
    """
    features {
      feature {
        key: "query_tokens"
        value { bytes_list { value: ["this", "is", "a", "relevant", "question"] } }
      }
    }""", tf.train.Example())
    
EXAMPLES = [
    text_format.Parse(
    """
    features {
      feature {
        key: "document_tokens"
        value { bytes_list { value: ["this", "is", "a", "relevant", "answer"] } }
      }
      feature {
        key: "relevance"
        value { int64_list { value: 5 } }
      }
    }""", tf.train.Example()),
    text_format.Parse(
        """
    features {
      feature {
        key: "document_tokens"
        value { bytes_list { value: ["irrelevant", "data"] } }
      }
      feature {
        key: "relevance"
        value { int64_list { value: 1 } }
      }
    }""", tf.train.Example()),
]
        
ELWC = input_pb2.ExampleListWithContext()
ELWC.context.CopyFrom(CONTEXT)
for example in EXAMPLES:
  example_features = ELWC.examples.add()
  example_features.CopyFrom(example)

print(ELWC)

# Store the paths to files containing training and test instances.
_TRAIN_DATA_PATH = "Data/Raw/tfrecords/train.tfrecords"
_TEST_DATA_PATH = "Data/Raw/tfrecords/test.tfrecords"

# Store the vocabulary path for query and document tokens.
_VOCAB_PATH = "Data/Raw/tfrecords/vocab.txt"

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 50

# The document relevance label.
_LABEL_FEATURE = "relevance"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1

# Learning rate for optimizer.
_LEARNING_RATE = 0.05

# Parameters to the scoring function.
_BATCH_SIZE = 32
_HIDDEN_LAYER_DIMS = ["64", "32", "16"]
_DROPOUT_RATE = 0.8
_GROUP_SIZE = 1  # Pointwise scoring.

# Location of model directory and number of training steps.
_MODEL_DIR = "Data/Modeling/ranking_model_dir_tfrank"
_NUM_TRAIN_STEPS = 15 * 1000

#Las columnas de características son abstracciones de TensorFlow que se utilizan para capturar información enriquecida sobre cada característica.
#Permiten transformaciones sencillas para una amplia gama de características sin procesar y para interactuar con los estimadores.

#En consonancia con nuestros formatos de entrada para la clasificación, como el formato ELWC,
#creamos columnas de características para las características de contexto y las características de ejemplo.

_EMBEDDING_DIMENSION = 20

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

#El lector de entrada lee los datos del almacenamiento persistente para producir tensores densos y dispersos en bruto del tipo apropiado para cada característica.
#Las características de ejemplo se representan mediante tensores tridimensionales (cuyas dimensiones corresponden a las consultas,
#los ejemplos y los valores de las características).
#Las características de contexto se representan mediante tensores bidimensionales 
#(cuyas dimensiones corresponden a las consultas y a los valores de las características).

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

#Transformaciones de características con transform_fn
#La función transform toma las características densas o dispersas en bruto del lector de entrada,
#aplica las transformaciones adecuadas para devolver representaciones densas para cada característica.

#Esto es importante antes de pasar estas características a una red neuronal, ya que las capas de redes neuronales suelen tomar características densas como entradas.

#La función de transformación maneja cualquier transformación de características personalizada definida por el usuario.
#Para manejar características dispersas, como los datos de texto, proporcionamos una función sencilla para crear incrustaciones compartidas,
#basadas en las columnas de características.

def make_transform_fn():
  def _transform_fn(features, mode):
    """Defines transform_fn."""
    context_features, example_features = tfr.feature.encode_listwise_features(
        features=features,
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns(),
        mode=mode,
        scope="transform_layer")

    return context_features, example_features
  return _transform_fn

#Interacciones entre características mediante scoring_fn
#A continuación, pasamos a la función de puntuación, que podría decirse que es el núcleo de un modelo de clasificación TF.
#La idea es calcular una puntuación de relevancia para un (conjunto de) par(es) consulta-documento.
#El modelo TF-Ranking utilizará datos de entrenamiento para aprender esta función.

#Aquí formulamos una función de puntuación utilizando una red de alimentación directa.
#La función toma las características de un único ejemplo (es decir, el par consulta-documento) y produce una puntuación de relevancia.
     
def make_score_fn():
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a group of documents."""
    with tf.compat.v1.name_scope("input_layer"):
      context_input = [
          tf.compat.v1.layers.flatten(context_features[name])
          for name in sorted(context_feature_columns())
      ]
      group_input = [
          tf.compat.v1.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns())
      ]
      input_layer = tf.concat(context_input + group_input, 1)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = input_layer
    cur_layer = tf.compat.v1.layers.batch_normalization(
      cur_layer,
      training=is_training,
      momentum=0.99)

    for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
        cur_layer,
        training=is_training,
        momentum=0.99)
      cur_layer = tf.nn.relu(cur_layer)
      cur_layer = tf.compat.v1.layers.dropout(
          inputs=cur_layer, rate=_DROPOUT_RATE, training=is_training)
    logits = tf.compat.v1.layers.dense(cur_layer, units=_GROUP_SIZE)
    return logits

  return _score_fn

#Pérdidas, métricas y clasificación Head
#Métricas de Evaluación
#Hemos proporcionado una implementación de varias métricas de evaluación populares de Recuperación de Información en la biblioteca de Ranking TF,
#que se muestran aquí.
#El usuario también puede definir una métrica de evaluación personalizada, como se muestra en la siguiente descripción

def eval_metric_fns():
  """Returns a dict from name to metric functions.

  This can be customized as follows. Care must be taken when handling padded
  lists.

  def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
  metric_fns["auc"] = _auc

  Returns:
    A dict mapping from metric name to a metric function with above signature.
  """
  metric_fns = {}
  metric_fns.update({
      f"metric/ndcg@{topn}": tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10]
  })

  return metric_fns

#Pérdidas de clasificación
#Proporcionamos varias funciones de pérdida de clasificación populares como parte de la biblioteca, que se muestran aquí.
#El usuario también puede definir una función de pérdida personalizada, similar a las de tfr.losses.

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.

_LOSS = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
loss_fn = tfr.losses.make_loss_fn(_LOSS)

#Clasificación Head
#En el flujo de trabajo de Estimator, Head es una abstracción que encapsula las pérdidas y las métricas correspondientes.
#Head interactúa fácilmente con el Estimator, necesitando que el usuario defina una función de puntuación y especifique las pérdidas y el cálculo de métricas.

optimizer = tf.compat.v1.train.AdagradOptimizer(
    learning_rate=_LEARNING_RATE)


def _train_op_fn(loss):
  """Defines train op used in ranking head."""
  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  minimize_op = optimizer.minimize(
      loss=loss, global_step=tf.compat.v1.train.get_global_step())
  train_op = tf.group([update_ops, minimize_op])
  return train_op


ranking_head = tfr.head.create_ranking_head(
      loss_fn=loss_fn,
      eval_metric_fns=eval_metric_fns(),
      train_op_fn=_train_op_fn)

#Reunir todo en un generador de modelos
#Ahora estamos listos para juntar todos los componentes anteriores y crear un Estimador que pueda utilizarse para entrenar y evaluar un modelo.

model_fn = tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          transform_fn=make_transform_fn(),
          group_size=_GROUP_SIZE,
          ranking_head=ranking_head)

def train_and_eval_fn():
  """Train and eval function used by `tf.estimator.train_and_evaluate`."""
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=1000)
  ranker = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=_MODEL_DIR,
      config=run_config)

  train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
  eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)
  eval_spec = tf.estimator.EvalSpec(
      name="eval",
      input_fn=eval_input_fn,
      throttle_secs=15)
  return (ranker, train_spec, eval_spec)

#! rm -rf "/tmp/ranking_model_dir"  # Clean up the model directory.
ranker, train_spec, eval_spec = train_and_eval_fn()
tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)

#%load_ext tensorboard
#%tensorboard --logdir="/tmp/ranking_model_dir" --port 12345



  