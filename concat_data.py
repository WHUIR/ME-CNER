import os


def concat_weibo():
    """
    CONCAT三个数据集
    :return:
    """
    if not os.path.exists('process_character/data_preprocess/weibo_raw_data.txt'):
        lines = []
        for file_suffix in ['train', 'test', 'dev']:
            with open('process_character/data_preprocess/weiboNER_2nd_conll.%s' % file_suffix, 'r',
                      encoding='utf-8') as f:
                for line in f:
                    lines.append(line)

        with open('process_character/data_preprocess/weibo_raw_data.txt', 'w', encoding='utf-8') \
                as f:
            f.writelines(lines)


def concat_msra():
    """
    CONCAT三个数据集
    :return:
    """
    if not os.path.exists('process_character/data_preprocess/msra_raw_data.txt'):
        lines = []
        for file_suffix in ['train', 'test', 'dev']:
            with open('process_character/data_preprocess/msra.%s' % file_suffix, 'r',
                      encoding='utf-8') as f:
                for line in f:
                    lines.append(line)

        with open('process_character/data_preprocess/msra_raw_data.txt', 'w', encoding='utf-8') \
                as f:
            f.writelines(lines)


if __name__ == '__main__':
    concat_weibo()
    concat_msra()
