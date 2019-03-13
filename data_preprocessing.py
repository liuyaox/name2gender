# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:52:45 2019
@author: liuyao8
"""

import numpy as np


def get_data_xy(data_file, header=True):
    """从原始数据中加工数据X和Y"""
    data_x, data_y = [], []
    with open(data_file, 'r') as fr:
        for line in fr:
            if header is True:
                header = False
            else:
                sample = line.strip().split(',')
                if len(sample) == 2:
                    data_x.append(sample[0])
                    gender = [0, 1] if sample[1] == '男' else [1, 0]  # 对于二分类，为什么不直接使用0和1,而是向量？ 方便扩展为多分类！
                    data_y.append(gender)
    max_name_length = max([len(name) for name in data_x])
    return max_name_length, data_x, data_y


def data_shuffle(data_x, data_y):
    """对数据进行shuffle"""
    shuffle_indices = np.random.permutation(np.arange(len(data_y)))
    data_x = np.array(data_x)[shuffle_indices]
    data_y = np.array(data_y)[shuffle_indices]
    return data_x, data_y


def get_vocabulary(names):
    """
    词汇表：character-level 而非word-level
    统计character字频并倒序排序获得index，构建词汇字典：<character, index> 后续会使用index来表示character
    """
    vocab = {}
    for name in names:
        for character in name:
            if character in vocab:
                vocab[character] += 1
            else:
                vocab[character] = 1
    vocab_list = [' '] + sorted(vocab, key=vocab.get, reverse=True)  # 按character字频倒序排列
    vocab_len = len(vocab_list)
    vocabulary = dict([(x, y) for (y, x) in enumerate(vocab_list)])     # character词汇表：排列rank作为character的index
    return vocab_len, vocabulary


def name_to_vec(name, vocabulary, max_name_length):
    """基于词汇表vocab，把姓名name转化为向量"""
    name_vec = [vocabulary.get(character, 0) for character in name]  # 姓名转化为每个character对应的index
    paddings = [0] * (max_name_length - len(name_vec))               # 小于向量长度的部分用0来padding
    name_vec.extend(paddings)                         
    return name_vec
