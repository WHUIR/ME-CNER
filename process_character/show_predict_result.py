import os
import pickle

import numpy as np

from process_character.constants import Dataset
from process_character.dicts import label_dict
from train_ner.model import calculate_precision_recall_f1, TIME_STEPS


def print_predicted_data(predict, ground_truth, dataset=Dataset.WEIBO):
    p, r, f, cm = calculate_precision_recall_f1(
        predict,
        ground_truth,
        (6, 6) if dataset == Dataset.WEIBO else (28, 28)
    )

    print('precision=%s' % p)
    print('recall=%s' % r)
    print('f1=%s' % f)
    print(cm)
    with open('process_character/out/test.pkl' 'rb') as f:
        test = pickle.load(f)

    predict = np.argmax(predict, predict.ndim - 1)
    ground_truth = np.argmax(ground_truth, ground_truth.ndim - 1)
    raw_chars_data = [row[1] for row in test]
    label_dict[0] = 'NIL'
    ground_truth = ground_truth.reshape(-1, TIME_STEPS)

    if not os.path.exists('process_character/error_analysis'):
        os.mkdir('process_character/error_analysis')
    with open('process_character/error_analysis/error_analyse.txt', 'w', encoding='utf-8') as f:
        for chars_row, predict_row, gt_row in zip(raw_chars_data, predict, ground_truth):
            for char, predict_label, gt_label in zip(chars_row, predict_row, gt_row):
                line = ''
                line += char
                line += '\t\t'
                line += label_dict[predict_label]
                line += '\t\t'
                line += label_dict[gt_label]
                line += '\n'
                f.write(line)
            f.write('\n')


def show_predict_data(predict_file, ground_truth_file):
    predict = np.load(predict_file)
    gt = np.load(ground_truth_file)
    print_predicted_data(predict, gt)


if __name__ == '__main__':
    show_predict_data('train_ner/predict.npy', 'train_ner/ground_truth.npy')
