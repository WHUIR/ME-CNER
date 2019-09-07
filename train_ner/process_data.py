from configparser import ConfigParser

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

from process_character.constants import Dataset

cf = ConfigParser()
cf.read(r'train_ner/train.cfg')
CHUCK_TAGS_NUM = int(cf.get('size', 'CHUCK_TAGS_NUM'))
MSRA_CHUCK_TAGS_NUM = int(cf.get('msra_size', 'CHUCK_TAGS_NUM'))
RAW_WORDS_COL = 0
CHAR_COL = 2
WORD_COL = 3
WORD_EMBEDDING_COL = 4
PINYIN_COL = 5
RADICAL_COL = 6
COMPONENT_COL = 7
LABEL_COL = 8
TONE_COL = 9
SYLLABLE_COL = 10
FAMILY_COL = 11
POSITION_COL = 12
IN_NAMES_COL = 13


def load_data(dataset):
    with open('process_character/out/{}train.pkl'.format(
            '' if dataset == Dataset.WEIBO else 'msra_'
    ), 'rb') as f:
        train = pickle.load(f)
    with open('process_character/out/{}test.pkl'.format(
            '' if dataset == Dataset.WEIBO else 'msra_'
    ), 'rb') as f:
        test = pickle.load(f)
    with open('process_character/out/{}dev.pkl'.format(
            '' if dataset == Dataset.WEIBO else 'msra_'
    ), 'rb') as f:
        dev = pickle.load(f)

    whole_data = (train, test, dev)
    sentence_maxlen = get_sentence_maxlen(whole_data)
    word_maxlen = get_word_maxlen(whole_data)

    train = _process_data(train, {
        'c': CHAR_COL,
        'w': WORD_COL,
        'rad': RADICAL_COL,
    }, LABEL_COL, {
                              'sentence': sentence_maxlen,
                              'word': word_maxlen,
                          }, dataset)
    test = _process_data(test, {
        'c': CHAR_COL,
        'w': WORD_COL,
        'rad': RADICAL_COL,
    }, LABEL_COL, {
                             'sentence': sentence_maxlen,
                             'word': word_maxlen,
                         }, dataset)
    dev = _process_data(dev, {
        'c': CHAR_COL,
        'w': WORD_COL,
        'rad': RADICAL_COL,
    }, LABEL_COL, {
                            'sentence': sentence_maxlen,
                            'word': word_maxlen,
                        }, dataset)

    return train, test, dev


def get_sentence_maxlen(data):
    maxlen = 0
    for data_chunk in data:
        x = [row[CHAR_COL] for row in data_chunk]
        maxlen = max(max(len(sentence) for sentence in x), maxlen)

    return maxlen


def get_word_maxlen(data):
    maxlen = 0
    for data_chunk in data:
        x = [row[RAW_WORDS_COL] for row in data_chunk]
        maxlen = max(max(len(word) for sentence in x for word in sentence), maxlen)

    return maxlen


def get_components_maxlen(data):
    maxlen = 0
    for data_chunk in data:
        x = [row[COMPONENT_COL] for row in data_chunk]  # [None, 83, 22, ?]
        maxlen = max(max(
            len(char) for sentence in x for char in sentence
        ), maxlen)

    return maxlen


def _process_data(data, x_col_dict, y_col, maxlen, dataset):
    x_c = [row[x_col_dict['c']] for row in data]
    x_w = [row[x_col_dict['w']] for row in data]
    x_rad = [row[x_col_dict['rad']] for row in data]
    y = [row[y_col] for row in data]

    y = pad_sequences(y, dtype='int', padding='post', maxlen=maxlen['sentence'], value=0)
    x_c = pad_sequences(x_c, dtype='int', padding='post', maxlen=maxlen['sentence'], value=0)
    x_w = pad_sequences(x_w, dtype='int', padding='post', maxlen=maxlen['sentence'], value=0)
    x_rad = pad_sequences(x_rad, dtype='int', padding='post', maxlen=maxlen['sentence'], value=0)

    new_y = []
    chunk_tags_num = CHUCK_TAGS_NUM if dataset == Dataset.WEIBO else MSRA_CHUCK_TAGS_NUM
    for row in y:
        current_row = []
        for label_index in row:
            one_hot_row = [0] * label_index + [1] + [0] * (chunk_tags_num - 1 - label_index)
            current_row.append(one_hot_row)
        new_y.append(current_row)

    y = np.array(new_y)
    print(y.shape)

    return x_c, x_w, x_rad, y
