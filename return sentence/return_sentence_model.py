#! -*- coding: utf-8 -*-
# return the token embeddings to the sentences utilizing network model

from bert4keras.backend import keras, search_layer, K
from keras.models import Model
import numpy as np
from bert4keras.tokenizers import Tokenizer
from keras.engine.topology import get_source_inputs
from keras.layers import Input, Dense
from sklearn.metrics import accuracy_score


data_path = 'datasets/lcqmc/lcqmc.train.data'
embedding_matrix = np.load('datasets/lcqmc/token_embedding/embedding_matrix.npy', allow_pickle=True)
tokenizer = Tokenizer(token_dict='../../chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


def convert_input_data(data_path):
    token_list, embedding_list, sentence_len = [], [], []
    train_data = load_data(data_path)
    for text in train_data:
        token_ids, segment_ids = tokenizer.encode(text[0], text[1], maxlen=128)
        for id in token_ids:
            embedding_list.extend(embedding_matrix[id])
        token_list.extend(token_ids)
        sentence_len.append(len(token_ids))
    print(len(embedding_list))
    print(len(token_list))
    print(sum(sentence_len))
    return embedding_list, token_list, sentence_len


def return_sentences(token_ids, length):
    sentences = []
    tokens = tokenizer.ids_to_tokens(token_ids)
    for len in length:
        sentence_pair = ''.join(tokens.pop(len))
        sentences.append(sentence_pair)
    return sentences


inputs = Input(shape=(10,))
x = Dense(100, activation='relu')(inputs)
x = Dense(100, activation='relu')(x)
output = Dense(21128, activation='softmax')(x)
model = Model(inputs, output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def evaluate(valid_x, valid_y):
    y_pred = model.predict(valid_x).argmax(axis=1)
    acc = accuracy_score(valid_y, y_pred)
    pre_result = return_sentences(y_pred, sentences_length)
    print(pre_result)
    return acc


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_x, valid_y)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
        test_acc = evaluate(test_x, test_y)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    train_x, train_y, sentences_length = convert_input_data(data_path)
    valid_x, valid_y, test_x, test_y = train_x, train_y, train_x, train_y
    evaluator = Evaluator()

    print('Training---------')
    model.fit(
        train_x,
        train_y,
        epochs=10,
        batch_size=10,
        callbacks=[evaluator]
    )
