#! -*- coding: utf-8 -*-
# return the token embeddings to the sentences utilizing network model
import numpy as np
from bert4keras.backend import keras
from keras.models import Model
from keras.layers import Input, Dense
from bert4keras.optimizers import Adam
from bert4keras.tokenizers import Tokenizer
from sklearn.metrics import accuracy_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tokenizer = Tokenizer(token_dict='C:/Users/admin/Desktop/generatot/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


def convert_data(data_path, test_flag=False):
    token_list, sentence_len = [], []
    train_data = load_data(data_path)
    for text in train_data:
        token_ids, segment_ids = tokenizer.encode(text[0], text[1], maxlen=128)
        token_list.extend(token_ids)
        if test_flag is True:
            sentence_len.append(len(token_ids))
    return np.array(token_list), sentence_len


def return_sentences(token_ids, length):
    sentences = []
    tokens = tokenizer.ids_to_tokens(token_ids)
    for len in length:
        sentence_pair = ''.join(tokens[:len])
        tokens = tokens[len:]
        sentences.append(sentence_pair)
    return sentences


inputs = Input(shape=(768,))
x = Dense(100, activation='gelu')(inputs)
x = Dense(100, activation='gelu')(x)
output = Dense(21128, activation='softmax')(x)
model = Model(inputs, output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def evaluate(x, y, sentences_length=None):
    y_pred = model.predict(x).argmax(axis=1)
    acc = accuracy_score(y, y_pred)
    if sentences_length is not None:
        pre_result = return_sentences(y_pred, sentences_length)
        print(pre_result[:10])
    return acc


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_x[:2000], valid_y[:2000])
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('weight/best_model.weights')
        test_acc = evaluate(test_x[:2000], test_y[:2000], test_len[:100])
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    train_x = np.load('train_embedding.npy', allow_pickle=True)
    valid_x = np.load('valid_embedding.npy', allow_pickle=True)
    test_x = np.load('delta_test_embedding.npy', allow_pickle=True)

    train_y, _ = convert_data('../examples/datasets/lcqm-data/lcqmc.train.data')
    valid_y, valid_len = convert_data('../examples/datasets/lcqm-data/lcqmc.dev.data')
    test_y, test_len = convert_data('../examples/datasets/lcqm-data/lcqmc.test.data', True)
    evaluator = Evaluator()
    model.load_weights('weight/best_model.weights')
    print('Training---------')
    model.fit(
        train_x,
        train_y,
        epochs=3,
        batch_size=42,
        callbacks=[evaluator]
    )
