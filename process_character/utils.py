from process_character.lib.langconv import Converter


def is_cn_or_digit(s):
    """
    字符是否属于汉字或数字
    """
    for c in s:
        if not ('\u4e00' <= c <= '\u9fff' or '0' <= c <= '9'):
            return False
    return True


def Traditional2Simplified(sentence):
    """
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    """
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def Simplified2Traditional(sentence):
    """
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    """
    sentence = Converter('zh-hant').convert(sentence)
    return sentence


if __name__ == '__main__':
    assert is_cn_or_digit('我')
    assert is_cn_or_digit('我213去')
    assert is_cn_or_digit('13')
    assert is_cn_or_digit('去')
    assert not is_cn_or_digit('a')
    assert not is_cn_or_digit('12.35')
