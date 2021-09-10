#! -*- coding: utf-8 -*-
# 这种方案没成功
# 输入：embeddings。输出：原始句子
# 预测：加干扰的测试集embedding

import glob
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import re
import json
import os

# 基本参数
maxlen = 256
batch_size = 8
# steps_per_epoch = 600
epochs = 3

# bert配置
config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'


def load_data(sentence_file):
    """加载数据
    单条格式：(embedding, raw_sentence)
    """
    D = []

    f_sentence = open(sentence_file, encoding='utf-8')
    for l in f_sentence:
        text1, text2, label = l.strip().split('\t')
        text = text1 + text2
        D.append(text)
    return D


# 加载数据集
train_embedding = np.load('examples/datasets/lcqmc/train_embedding.npy')
test_embedding = np.load('examples/datasets/lcqmc/test_embedding.npy')
train_sentence = load_data('examples/datasets/lcqmc/lcqmc.test.data')
test_sentence = load_data('examples/datasets/lcqmc/lcqmc.test.data')


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
# model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=2):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


def result():
    for s1 in test_embedding:
        fw = open('generate_result.txt', 'w', encoding='utf-8')
        fw.write(autotitle.generate(s1))


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        result()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_embedding,
        train_sentence,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[evaluator]
    )
