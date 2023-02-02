# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:06:31 2023

@author: corde
"""

import pathlib

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_text as tf_text
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format

example_list_with_context = {
    "context": {
        "query_tokens": ["this", "is", "a", "question"]
    },
    "examples": [
        {
            "document_tokens": ["this", "is", "a", "relevant", "answer"],
            "relevance": [4]
        },
        {
            "document_tokens": ["irrelevant", "data"],
            "relevance": [0]
        }
    ]
}

CONTEXT = text_format.Parse(
    """
    features {
      feature {
        key: "query_tokens"
        value { bytes_list { value: ["this", "is", "a", "question"] } }
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
        value { int64_list { value: 4 } }
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
        value { int64_list { value: 0 } }
      }
    }""", tf.train.Example()),
]

ELWC = input_pb2.ExampleListWithContext()
ELWC.context.CopyFrom(CONTEXT)

for example in EXAMPLES:

    example_features = ELWC.examples.add()
    example_features.CopyFrom(example)

print(ELWC)

serialized_elwc = ELWC.SerializeToString()
print(serialized_elwc)

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

parse_elwc(serialized_elwc)







