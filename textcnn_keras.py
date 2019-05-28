#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:53:18 2019
@author: liuyao8
"""

import numpy as np
from functools import reduce
import tensorflow as tf
from keras.layers import Input, Embedding, Dropout, Dense, Conv1D, BatchNormalization
from keras.layers import Concatenate, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import tokenizer_from_json
from keras.models import load_model
from keras.utils import plot_model
from keras.initializers import Constant

from data_preprocessing import get_data_xy, data_shuffle, get_vocabulary, name_to_vec


# 常量和基本配置
name_dataset = './data/name.csv'
pretrained_file_public = ''
emb_dim_public = 200
pretrained_file_title = ''
emb_dim_title = 10
structured_dim = 150  # 结构化向量的维度


max_name_length, data_x, data_y = get_data_xy(name_dataset, header=True)
max_name_length = 8
data_x, data_y = data_shuffle(data_x, data_y)   
vocab_len, vocabulary = get_vocabulary(data_x)
data_x_vec = [name_to_vec(name, vocabulary, max_name_length) for name in data_x]


# 训练/测试数据
train_pword1, train_pword2, train_structured = None, None, None
train_y = None
test_pword1, test_pword2, test_structured = None, None, None
test_y = None



input_size = max_name_length
num_classes = 2
batch_size = 64
num_batch = len(data_x_vec) // batch_size


def get_word2vector(pretrained_file, emb_dim):
    """ Create dictionary mapping word to embedding using pretrained word embeddings """
    word2vector = {}
    with open(pretrained_file, 'r', encoding='utf-8') as fr:
        fr.readline()                        # Drop line 1
        for line in fr:
            values = line.strip().split()
            try:
                word = values[0]
                vector = np.asarray(values[1: emb_dim+1], dtype='float32')
                word2vector[word] = vector
            except ValueError as e:
                pass
    return word2vector

word2vector_public = get_word2vector(pretrained_file_public, emb_dim_public)
word2vector_title  = get_word2vector(pretrained_file_title,  emb_dim_title)


def get_emb_layer(initializer='constant', vocabulary=None, word2vector=None, emb_dim=128, name=None):
    """ Create Embedding Layer Using Random Initialization or Pretrained Word Embeddings """
    # Using random initialization
    if initializer != 'constant':
        emb_layer = Embedding(vocab_len, emb_dim, embeddings_initializer=initializer, name=name, trainable=True) # 一般取uniform
    # Using pre-trained word embeddings
    elif word2vector is not None and vocabulary is not None:
        emb_dim = word2vector.get('我').shape[0]
        emb_matrix = np.zeros((vocab_len, emb_dim))     # 此处难道不应该是vocab_len+1么？！？
        for word, index in vocabulary.items():          # vocabulary的index从1开始，0留给谁的？！？
            if index < vocab_len:
                if word in word2vector:
                    vector = word2vector.get(word)
                else:
                    # 若word不存在，则取所有character vector的均值作为embedding
                    vectors = [word2vector.get(x, np.zeros(emb_dim)) for x in list(word)]
                    vector = reduce(lambda x, y: x + y, vectors) / len(vectors)
                if vector is not None:
                    emb_matrix[index, :] = vector       # index=0没有赋值，留给谁？！？
        emb_layer = Embedding(vocab_len, emb_dim, embeddings_initializer=Constant(emb_matrix), name=name, trainable=False)
    else:
        print('ERROR! There is no vocabulary or word2vector!')
    return emb_layer

# 4 Embedding Layers
# public is for both pword1 and pword2, since trainable=False ?    For public, shape of result is (None, max_name_length, emb_dim)
emb_layer_public  = get_emb_layer('constant', vocabulary, word2vector_public, name='emb_public')
emb_layer_title   = get_emb_layer('constant', vocabulary, word2vector_title, name='emb_title')
emb_layer_random1 = get_emb_layer('uniform', emb_dim=128, name='emb_random1')     # For pword1, since its trainable=True ?
emb_layer_random2 = get_emb_layer('uniform', emb_dim=128, name='emb_random2')     # For pword2


def TextCNN(X, fsizes=[2, 3, 4], units=32, name=None):
    """ TextCNN: (Embedding) -> (Conv1D -> MaxPooling) * n -> Concatenate -> Dense """
    Xs = []
    for fsize in fsizes:
        Xi = Conv1D(128, fsize, activation='relu')(X)       # (None, maxlen-fsize+1, 128)
        Xi = GlobalMaxPooling1D()(Xi)                       # (None, 128)
        Xs.append(Xi)
    X = Concatenate(axis=-1, name=name)(Xs)                 # (None, 128*3)
    X = Dropout(0.5)(X)
    X = Dense(units=units, activation='relu')(X)            # (None, units)
    return X


def Model_TextCNN(input_shapes, emb_layers):
    """ TextCNN-based Model with 3 Inputs and 1 Output """
    # 3 Inputs
    pword1 = Input(input_shapes[0], dtype='int32', name='pword1')
    pword2 = Input(input_shapes[1], dtype='int32', name='pword2')
    structured = Input(input_shapes[2], dtype='float32', name='structured')   # float32 ?
    # 4 or 6 Embedded Inputs
    pword1_emb_list = [layer(pword1) for layer in emb_layers[0]]
    pword2_emb_list = [layer(pword2) for layer in emb_layers[1]]
    # 2 Concatenated Inputs
    pword1_embs = Concatenate(axis=-1, name='pword1_embs')(pword1_emb_list)
    pword2_embs = Concatenate(axis=-1, name='pword2_embs')(pword2_emb_list)
    # 2 TextCNNs
    pword1_textcnn = TextCNN(pword1_embs, fsizes=[2, 3, 4], units=32, name='pword1_textcnn')
    pword2_textcnn = TextCNN(pword2_embs, fsizes=[2, 3, 4], units=32, name='pword2_textcnn')
    # All Concatenated
    X = Concatenate(axis=-1, name='x_concat')([pword1_textcnn, pword2_textcnn, structured])
    X = BatchNormalization()(X)                         # Necessary ?
    X = Dropout(0.5)(X)                                 # Necessary ?
    X = Dense(units=32, activation='relu')(X)
    X = Dense(units=1, activation='sigmoid')(X)
    return Model([pword1, pword2, structured], X)

input_shapes = [(max_name_length, ), (max_name_length, ), (structured_dim, )]
emb_layers = [(emb_layer_public, emb_layer_title, emb_layer_random1), (emb_layer_public, emb_layer_title, emb_layer_random2)]
model = Model_TextCNN(input_shapes, emb_layers)
model.summary()
plot_model(model)


# 模型配置
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # patience可以稍微大点
optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# 模型训练、评估和应用
model.fit([train_pword1, train_pword2, train_structured], train_y, epoches=50, batch_size=32, callbacks=[early_stopping])
accuracy = model.evaluate([test_pword1, test_pword2, test_structured], test_y)[1]
pred = model.predict([[test_pword1, test_pword2, test_structured]])
pred2 = pred.argmax(1)
