#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:53:18 2019
@author: liuyao8
"""

import numpy as np
from functools import reduce
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, Flatten, Conv1D
from keras.layers import Concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import tokenizer_from_json
from keras.models import load_model
from keras.initializers import Constant

from data_preprocessing import get_data_xy, data_shuffle, get_vocabulary, name_to_vec


# 常量和基本配置
name_dataset = './data/name.csv'

max_name_length, data_x, data_y = get_data_xy(name_dataset, header=True)
max_name_length = 8
data_x, data_y = data_shuffle(data_x, data_y)   
vocab_len, vocabulary = get_vocabulary(data_x)
data_x_vec = [name_to_vec(name, vocabulary, max_name_length) for name in data_x]


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


def get_embedding_layer(initializer='constant', vocabulary=None, word2vector=None, emb_dim=None):
    """ Create Embedding Layer Using Random Initialization or Pretrained Word Embeddings """
    # Using random initialization
    if initializer != 'constant':
        emb_layer = Embedding(vocab_len, emb_dim, embeddings_initializer=initializer, trainable=True) # 一般取uniform
    # Using pre-trained word embeddings
    elif word2vector is not None and vocabulary is not None:
        emb_dim = word2vector.get('a').shape[0]
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
        emb_layer = Embedding(vocab_len, emb_dim, embeddings_initializer=Constant(emb_matrix), trainable=False)
    else:
        print('ERROR! There is no vocabulary or word2vector!')
    return emb_layer


def TextCNN(input_shape, emb_layer, fsizes=[2, 3, 4], units=32):
    """ TextCNN: Embedding -> (Conv1D -> MaxPooling) * n -> Concatenate -> Dense """
    X0 = Input(input_shape, dtype='int32')
    X = emb_layer(X0)                                       # (None, maxlen, emb_dim)
    Xs = []
    for fsize in fsizes:
        Xi = Conv1D(128, fsize, activation='relu')(X)       # (None, maxlen-fsize+1, 128)
        Xi = GlobalMaxPooling1D()(Xi)                       # (None, 128)
        Xs.append(Xi)
    X = Concatenate(axis=-1)(Xs)                            # (None, 128*3)
    X = Dropout(0.5)(X)
    X = Dense(units=units, activation='relu')(X)            # (None, units)
#    X = Dense(units=1, activation='sigmoid')(X)             # (None, 1)
#    model = Model(X0, X)
    return X




inputs = Input(shape=(maxlen,))
inter = Embedding(output_dim=embedding_dims, input_dim=max_words, input_length=maxlen)(inputs)
inter = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(inter)
inter = GlobalMaxPooling1D()(inter)
inter = Dense(10, activation='relu')(inter)

price_input = Input(shape=(1,))

comb_inter = concatenate([inter, price_input])

comb_inter = BatchNormalization()(comb_inter)
outputs = Dense(n_cate, activation='softmax')(comb_inter)

model2 = Model(inputs=[inputs, price_input], outputs=outputs)


model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
model2.fit([x, price_train], train_labels, epochs=20, batch_size=32)
accuracy = model2.evaluate([x_test, price_test], test_labels)[1]
y_pred2 = model2.predict([x_test, price_test])
y_predict2 = y_pred2.argmax(1)
