#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:53:18 2019
@author: liuyao8
"""

import tensorflow as tf
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

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)


# 模型定义: Embedding -> (Conv1d -> MaxPooling) * 3 -> Dropout -> FC
def TextCNN_TF(vocab_len, embedding_dim=128, num_filters=128):
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
            filter_shape = [filter_size, embedding_dim, 1, num_filters]             # 长度为embedding_dim，只上下滑动
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))          # 随机初始化W  暗含了filter信息
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))                  # 初始化b=0.1
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")  # Embedding输入CNN  conv = conv2d(W * e)
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
def train(epoches=201):
    output = TextCNN_TF(vocab_len)
 
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
                batch_x = data_x_vec[batch_i: batch_j]
                batch_y = data_y[batch_i: batch_j]
                _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
                if i % 500 == 0:
                    print(epoch, i, loss_)
                    
            if epoch % 20 == 0:
                saver.save(sess, "./model/name2gender.model", global_step=epoch)   # 保存模型


# 模型应用
def predict(name_list):
    # 恢复前一次训练
    output = TextCNN_TF(vocab_len)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")
        
        predictions = tf.argmax(output, 1)
        name_vec = [name_to_vec(name, vocabulary, max_name_length) for name in name_list]
        res = sess.run(predictions, {X: name_vec, dropout_keep_prob: 1.0})
 
        for i, name in enumerate(name_list):
            print(name, '女' if res[i] == 0 else '男')


tf.reset_default_graph()
train(100)
predict(["白富美", "高帅富", "王婷婷", "田野"])
