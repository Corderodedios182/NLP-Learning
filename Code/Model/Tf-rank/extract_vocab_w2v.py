# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:26:43 2023

@author: corde
"""

import os

os.chdir("C:/Users/corde/OneDrive/Documents/Model-20230131T172415Z-001/Model/Code/DataPrep/utils/models/w2v")

#Extraer vocab del w2v

import gensim

model_w2v = gensim.models.word2vec.Word2Vec.load(fname='model_w2v.model')

dir(model_w2v)

dir(model_w2v.wv)

vocab = model_w2v.wv.index_to_key

type(vocab)

with open('vocab_spanish.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in vocab))


