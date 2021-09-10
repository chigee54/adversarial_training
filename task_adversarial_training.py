#! -*- coding:utf-8 -*-
# 通过对抗训练增强模型的泛化性能
# 比CLUE榜单公开的同数据集上的BERT base的成绩高2%
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/7234
# 适用于Keras 2.3.1

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense, Layer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from itertools import chain
import keras_tuner as kt
# from tensorflow import keras

num_classes = 2
maxlen = 128
batch_size = 10

# BERT base
config_path = '../../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/lcqm-data/lcqmc.train.data')
valid_data = load_data('datasets/lcqm-data/lcqmc.dev.data')
test_data = load_data('datasets/lcqm-data/lcqmc.test.data')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def build_model(hp):
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)
    # output = EpsilonLayer()(output)

    model = keras.models.Model(bert.model.input, output)
    # model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(2e-5),
        metrics=['sparse_categorical_accuracy'],
    )


    # class EpsilonLayer(Layer):
    #
    #     def __init__(self, **kwargs):
    #         super(EpsilonLayer, self).__init__(**kwargs)
    #
    #     def build(self, input_layer):
    #         # 为该层创建一个可训练的权重
    #         self.epsilon = self.add_weight(name='epsilon',
    #                                       shape=(1, ),
    #                                       initializer='uniform',
    #                                       trainable=True)
    #         super(EpsilonLayer, self).build(input_layer)  # 一定要在最后调用它
    #
    #     def call(self, input_layer):
    #         return adversarial_training(model, 'Embedding-Token', self.epsilon)


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
            delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
            K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
            outputs = old_train_function(inputs)  # 梯度下降
            K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
            return outputs

        model.train_function = train_function  # 覆盖原训练函数


    # 写好函数后，启用对抗训练只需要一行代码
    adversarial_training(model, 'Embedding-Token', epsilon=0.1*hp.Int('epsilon', 1, 10, step=1))


tuner = kt.Hyperband(
        build_model, objective='sparse_categorical_accuracy', max_epochs=3, hyperband_iterations=2
    )


def evaluate(data):
    y_true_label, y_pred_label = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        y_true_label.append(list(y_true))
        y_pred_label.append(list(y_pred))
    y_true_label = list(chain.from_iterable(y_true_label))
    y_pred_label = list(chain.from_iterable(y_pred_label))
    f1 = f1_score(y_true_label, y_pred_label, average='binary')
    precision = precision_score(y_true_label, y_pred_label)
    recall = recall_score(y_true_label, y_pred_label)
    acc = accuracy_score(y_true_label, y_pred_label)
    return acc, f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, _, _, _ = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
        test_acc, f1, precision, recall = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f, test_f1_score: %.5f, test_precision: %.5f, test_recall: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc, f1, precision, recall)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    tuner.search(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=3,
        callbacks=[evaluator]
    )

    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
else:

    model.load_weights('best_model.weights')
