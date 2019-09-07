import os
import pickle

import jieba

from process_character.constants import Dataset

comps = []
radicals = []
characters = []
phrases = []
syllables = []


def load_radical_indexes():
    if not os.path.exists('process_character/data_preprocess/radical_indexes.txt'):
        with open('process_character/data_preprocess/radical.txt', 'r', encoding='utf-8') as f:
            for line in f:
                radicals.extend(line.split())

        with open('process_character/data_preprocess/radical_indexes.txt', 'w',
                  encoding='utf-8') as f:
            i = 1
            for radical in radicals:
                f.write('%s %s\n' % (i, radical))
                i = i + 1

    radical_dict = {}
    with open('process_character/data_preprocess/radical_indexes.txt', 'r', encoding='utf-8') as f:
        for line in f:
            index_radical_pair = line.split()
            radical_dict[int(index_radical_pair[0])] = index_radical_pair[1]
            radical_dict[index_radical_pair[1]] = int(index_radical_pair[0])

    return radical_dict


def load_character_indexes():
    char_index_dict = {}
    char_index_count = 1
    with open('process_character/data_preprocess/weibo_raw_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            char, _ = line.split()
            char = char[0]
            if char not in char_index_dict:
                char_index_dict[char] = char_index_count
                char_index_dict[char_index_count] = char
                char_index_count += 1
            characters.append(line[0])
    return char_index_dict


def load_phrase_indexes():
    if not os.path.exists('process_character/data_preprocess/phrase_indexes.txt'):

        with open('process_character/data_preprocess/weibo_raw_data.txt', 'r', encoding='utf-8') \
                as f:
            phrase_list = []
            sentence = []
            for line in f:
                if not line.strip():
                    words = list(jieba.cut(''.join(sentence)))
                    phrase_list.extend(words)
                    sentence = []
                    continue

                char, _ = line.split()
                char = char[0]
                sentence.append(char)

        phrase_set = list(set(phrase_list))
        phrase_set.sort(key=phrase_list.index)
        with open('process_character/data_preprocess/phrase_indexes.txt', 'w',
                  encoding='utf-8') as f:
            i = 1
            for phrase in phrase_set:
                if not phrase:
                    continue
                f.write('%s %s\n' % (i, phrase))
                i = i + 1

    phrase_dict = {}
    with open('process_character/data_preprocess/phrase_indexes.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.split():
                continue

            index_phrase_pair = line.split()
            phrase_dict[int(index_phrase_pair[0])] = index_phrase_pair[1]
            phrase_dict[index_phrase_pair[1]] = int(index_phrase_pair[0])

    return phrase_dict


def load_msra_character_indexes():
    char_index_dict = {}
    char_index_count = 1
    with open('process_character/data_preprocess/msra_raw_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            char, _ = line.split()
            char = char[0]
            if char not in char_index_dict:
                char_index_dict[char] = char_index_count
                char_index_dict[char_index_count] = char
                char_index_count += 1
            characters.append(line[0])
    return char_index_dict


def load_msra_phrase_indexes():
    if not os.path.exists('process_character/data_preprocess/msra_phrase_indexes.txt'):

        with open('process_character/data_preprocess/msra_raw_data.txt', 'r', encoding='utf-8') \
                as f:
            phrase_list = []
            sentence = []
            for line in f:
                if not line.strip():
                    words = list(jieba.cut(''.join(sentence)))
                    phrase_list.extend(words)
                    sentence = []
                    continue

                char, _ = line.split()
                char = char[0]
                sentence.append(char)

        phrase_set = list(set(phrase_list))
        phrase_set.sort(key=phrase_list.index)
        with open('process_character/data_preprocess/msra_phrase_indexes.txt', 'w',
                  encoding='utf-8') as f:
            i = 1
            for phrase in phrase_set:
                if not phrase:
                    continue
                f.write('%s %s\n' % (i, phrase))
                i = i + 1

    phrase_dict = {}
    with open('process_character/data_preprocess/msra_phrase_indexes.txt', 'r', encoding='utf-8') \
            as f:
        for line in f:
            if not line.split():
                continue

            index_phrase_pair = line.split()
            phrase_dict[int(index_phrase_pair[0])] = index_phrase_pair[1]
            phrase_dict[index_phrase_pair[1]] = int(index_phrase_pair[0])

    return phrase_dict


def load_label_indexes(dataset_type):
    label_dict = {}
    with open('process_character/data_preprocess/%slabel_indexes.txt' %
              ('msra_' if dataset_type == Dataset.MSRA else ''),
              'r', encoding='utf-8') as f:
        for line in f:
            index_label_pair = line.split()
            label_dict[int(index_label_pair[0])] = index_label_pair[1]
            label_dict[index_label_pair[1]] = int(index_label_pair[0])

    return label_dict
