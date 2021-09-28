#! -*- coding: utf-8 -*-

import numpy as np
from bert4keras.tokenizers import Tokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

embedding_matrix = np.load('embedding_matrix.npy', allow_pickle=True)
train_data_path = '../examples/datasets/lcqm-data/lcqmc.train.data'
valid_data_path = '../examples/datasets/lcqm-data/lcqmc.dev.data'
test_data_path = '../examples/datasets/lcqm-data/lcqmc.test.data'
tokenizer = Tokenizer(token_dict='C:/Users/admin/Desktop/generatot/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


def convert_data(data_path):
    embedding_list = []
    train_data = load_data(data_path)
    for text in train_data:
        token_ids, segment_ids = tokenizer.encode(text[0], text[1], maxlen=128)
        for id in token_ids:
            embedding_list.append(embedding_matrix[id])
    return np.array(embedding_list)


# np.save('train_embedding', convert_data(train_data_path))
# np.save('delta_valid_embedding', convert_data(valid_data_path))
np.save('test_embedding', convert_data(test_data_path))
