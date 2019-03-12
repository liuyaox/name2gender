#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:53:18 2019
@author: liuyao8
"""

import tensorflow as tf
import numpy as np


# 0. 常量和配置
name_dataset = './data/name.csv'


# 1. 功能函数
def get_train_xy(data_file, header=True):
    """从原始数据中加工训练数据X和Y"""
    train_x, train_y = [], []
    with open(data_file, 'r') as fr:
        for line in fr:
            if header is True:
                header = False
            else:
                sample = line.strip().split(',')
                if len(sample) == 2:
                    train_x.append(sample[0])
                    gender = [0, 1] if sample[1] == '男' else [1, 0]  # 对于二分类，为什么不直接使用0和1,而是向量？ 方便扩展为多分类！
                    train_y.append(gender)
    max_name_length = max([len(name) for name in train_x])
    return max_name_length, train_x, train_y
 
max_name_length, train_x, train_y = get_train_xy(name_dataset, header=True)
max_name_length = 8


def data_shuffle(train_x, train_y):
    """对数据进行shuffle"""
    shuffle_indices = np.random.permutation(np.arange(len(train_y)))
    train_x = train_x[shuffle_indices]
    train_y = train_y[shuffle_indices]
    return train_x, train_y

# train_x, train_y = data_shuffle(train_x, train_y)   


def get_vocab(names):
    """
    词汇表：character-level 而非word-level
    统计character字频并倒序排序获得index，构建词汇字典：<character, index> 后续会使用index来表示character
    """
    vocabulary = {}
    for name in names:
        for character in name:
            if character in vocabulary:
                vocabulary[character] += 1
            else:
                vocabulary[character] = 1
    vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)  # 按character字频倒序排列
    vocab_len = len(vocabulary_list)
    vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])     # character词汇表：排列rank作为character的index
    return vocab_len, vocab

vocab_len, vocab = get_vocab(train_x)


def name_to_vec(name, vocab):
    """基于词汇表vocab，把姓名name转化为向量"""
    name_vec = [vocab.get(character, 0) for character in name]  # 姓名转化为每个character对应的index
    paddings = [0] * (max_name_length - len(name_vec))          # 小于向量长度的部分用0来padding
    name_vec.extend(paddings)                         
    return name_vec

train_x_vec = [name_to_vec(name, vocab) for name in train_x]


#######################################################

# 常量和基本配置
input_size = max_name_length
num_classes = 2
batch_size = 64
num_batch = len(train_x_vec) // batch_size

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)


# 模型定义: Embedding -> (CNN -> MaxPooling) * 3 -> Dropout -> FC
def neural_network(vocab_len, embedding_dim=128, num_filters=128):
    # Embedding
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([vocab_len, embedding_dim], -1.0, 1.0))   # 随机初始化维度为(vocab_len, embedding_dim)的W
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # TextCNN: (CNN -> MaxPooling) * 3
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):                                  # 3个CNN通道并行
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_dim, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))          # 随机初始化W  暗含了filter信息
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))                  # 初始化b=0.1
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")  # 输入Embedding输入至CNN  conv = conv2d(W * e)
            h = tf.nn.relu(tf.nn.bias_add(conv, b))                                                 # h = relu(conv + b)
            pooled = tf.nn.max_pool(h, ksize=[1, input_size-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID') # pooled = maxpool(h)
            pooled_outputs.append(pooled)
 
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)                       # 合并3个CNN通道的结果
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    
    # Dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    
    # FC
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)                  # output = W * h_drop + b

    return output


# 模型训练
def neural_network_train(epoches=201, dropout_keep_prob=0.5):
    output = neural_network(vocab_len)
 
    # 定义优化器、损失、梯度和优化operator
    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
 
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            for i in range(num_batch):
                batch_i, batch_j = i * batch_size, (i + 1) * batch_size
                batch_x = train_x_vec[batch_i: batch_j]
                batch_y = train_y[batch_i: batch_j]
                _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: dropout_keep_prob})
                print(epoch, i, loss_)
                
            if epoch % 200 == 0:
                saver.save(sess, "name2gender.model", global_step=epoch)   # 保存模型
 
neural_network_train(50, 0.5)


# 模型应用
def neural_network_predict(name_list):
	x = []
	for name in name_list:
		name_vec = []
		for word in name:
			name_vec.append(vocab.get(word))
		while len(name_vec) < max_name_length:
			name_vec.append(0)
		x.append(name_vec)
 
	output = neural_network(vocab_len)
 
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		# 恢复前一次训练
		ckpt = tf.train.get_checkpoint_state('.')
		if ckpt != None:
			print(ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("没找到模型")
 
		predictions = tf.argmax(output, 1)
		res = sess.run(predictions, {X:x, dropout_keep_prob:1.0})
 
		i = 0
		for name in name_list:
			print(name, '女' if res[i] == 0 else '男')
			i += 1
 
neural_network_predict(["白富美", "高帅富", "王婷婷", "田野"])
