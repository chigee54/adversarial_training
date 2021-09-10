#! -*- coding: utf-8 -*-
# 句向量化: 训练集为无干扰，测试集为加干扰

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense, Layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from keras.models import Model
from bert4keras.optimizers import Adam

# BERT base
config_path = '../../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../chinese_L-12_H-768_A-12/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=2,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)
# output = EpsilonLayer()(output)

model = Model(bert.model.input, output)
# model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', epsilon=0.5)


def load_data(sentence_file):
    """加载数据
    """
    D = []
    f_sentence = open(sentence_file, encoding='utf-8')
    for l in f_sentence:
        text1, text2, label = l.strip().split('\t')
        text = text1 + text2
        D.append(text)
    return D


def predict(texts):
    """句子列表转换为句向量
    """
    batch_token_ids, batch_segment_ids = [], []
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=512)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    outputs = model.predict([batch_token_ids, batch_segment_ids])
    return outputs


def convert(data):
    """转换所有样本
    """
    embeddings = []
    for texts in tqdm(data, desc=u'向量化'):
        outputs = predict(texts)
        embeddings.append(outputs)
    embeddings = sequence_padding(embeddings)
    return embeddings


if __name__ == '__main__':

    # data_train = './datasets/lcqm-data/lcqmc.train.data'
    data_test = './datasets/lcqmc/lcqmc.test.data'

    # train_embedding_npy = './datasets/lcqmc/train_embedding'
    test_embedding_npy = './datasets/lcqmc/test_embedding'

    # embeddings_train = convert(load_data(data_train))
    embeddings_test = convert(load_data(data_test))

    # np.save(train_embedding_npy, embeddings_train)
    np.save(test_embedding_npy, embeddings_test)

    # print(u'输出路径：%s.npy' % train_embedding_npy)
    print(u'输出路径：%s.npy' % test_embedding_npy)
