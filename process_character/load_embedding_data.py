import os
import numpy as np
from process_character.dicts import phrase_dict, character_dict


def load_word_embedding():
    if os.path.exists('process_character/data_preprocess/embedding_data.txt'):
        print('load existing word embedding...')
        return

    embedding_dict = {}
    embedding_data = []
    hit = 0
    miss = 0
    error = 0
    phrase_list = []  # 按出现顺序的所有词
    phrase_dict_len = len(phrase_dict.keys()) // 2
    for i in range(1, phrase_dict_len + 1):
        phrase_list.append(phrase_dict[i])

    if not os.path.exists('process_character/data_preprocess/Tencent_AILab_ChineseEmbedding.txt'):
        raise RuntimeError('Failed to find TencentEmbedding data, please read the README and '
                           'download it.')

    print('load tencent embedding...')
    with open('process_character/data_preprocess/Tencent_AILab_ChineseEmbedding.txt', 'r',
              encoding='utf-8') as f:
        print('processing word embedding...')
        n_word, n_dim = f.readline().split()
        while True:
            try:
                line = f.readline()
                if not line.strip():
                    break
                else:
                    word, vector = line.split(' ', maxsplit=1)  # '科技', '-0.123 0.235 ...'
                    embedding_dict[word] = vector
            except:
                error += 1

    print('saving word embedding data...')
    for phrase in phrase_list:
        vector = embedding_dict.get(phrase)
        if vector:
            hit += 1
        else:
            miss += 1
        row = [phrase]
        row.extend(vector.split() if vector else np.random.randn(int(n_dim)))
        embedding_data.append(' '.join([str(_) for _ in row]))

    print('%s different words in your dataset' % len(phrase_list))
    print('%s words in Tencent embeddings' % n_word)
    print('%s dims' % n_dim)
    print('%s hits' % hit)
    print('%s miss' % miss)
    print('%s errors' % error)

    with open('process_character/data_preprocess/embedding_data.txt', 'w', encoding='utf-8') as \
            f:
        for line in embedding_data:
            f.write(line + '\n')


def load_char_embedding():
    if os.path.exists('process_character/data_preprocess/char_embedding_data.txt'):
        print('load existing char embedding...')
        return

    embedding_dict = {}
    embedding_data = []
    hit = 0
    miss = 0
    error = 0
    char_list = []  # 按出现顺序的所有char
    char_dict_len = len(character_dict.keys()) // 2
    for i in range(1, char_dict_len + 1):
        char_list.append(character_dict[i])

    if not os.path.exists('process_character/data_preprocess/Tencent_AILab_ChineseEmbedding.txt'):
        raise RuntimeError('Failed to find TencentEmbedding data, please read the README and '
                           'download it.')

    with open('process_character/data_preprocess/Tencent_AILab_ChineseEmbedding.txt', 'r',
              encoding='utf-8') as f:
        print('processing char embedding...')
        n_char, n_dim = f.readline().split()
        while True:
            try:
                line = f.readline()
                if not line.strip():
                    break
                else:
                    char, vector = line.split(' ', maxsplit=1)  # '科', '-0.123 0.235 ...'
                    embedding_dict[char] = vector
            except:
                error += 1

    for character in char_list:
        vector = embedding_dict.get(character)
        if vector:
            hit += 1
        else:
            miss += 1
        row = [character]
        row.extend(vector.split() if vector else np.random.randn(int(n_dim)))
        embedding_data.append(' '.join([str(_) for _ in row]))

    print('%s different chars in your dataset' % len(char_list))
    print('%s in Tencent embeddings' % n_char)
    print('%s dims' % n_dim)
    print('%s hits' % hit)
    print('%s miss' % miss)
    print('%s errors' % error)

    with open('process_character/data_preprocess/char_embedding_data.txt', 'w', encoding='utf-8') \
            as f:
        for line in embedding_data:
            f.write(line + '\n')


def run():
    load_word_embedding()
    load_char_embedding()
    print('embedding process finished')


if __name__ == '__main__':
    run()
