# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:52:45 2019
@author: liuyao8
"""

import numpy as np


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


def data_shuffle(train_x, train_y):
    """对数据进行shuffle"""
    shuffle_indices = np.random.permutation(np.arange(len(train_y)))
    train_x = np.array(train_x)[shuffle_indices]
    train_y = np.array(train_y)[shuffle_indices]
    return train_x, train_y


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


def name_to_vec(name, vocab, max_name_length):
    """基于词汇表vocab，把姓名name转化为向量"""
    name_vec = [vocab.get(character, 0) for character in name]  # 姓名转化为每个character对应的index
    paddings = [0] * (max_name_length - len(name_vec))          # 小于向量长度的部分用0来padding
    name_vec.extend(paddings)                         
    return name_vec
