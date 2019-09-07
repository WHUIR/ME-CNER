from process_character.constants import Dataset
from process_character.load_indexes import \
    load_radical_indexes, \
    load_character_indexes, \
    load_phrase_indexes, \
    load_msra_phrase_indexes, \
    load_label_indexes, \
    load_msra_character_indexes

char2radical = {}
radical_dict = {}
character_dict = {}
msra_character_dict = {}
phrase_dict = {}
msra_phrase_dict = {}
label_dict = {}
nam_nom_label_dict = {}
msra_label_dict = {}

if not radical_dict:
    radical_dict = load_radical_indexes()

if not character_dict:
    character_dict = load_character_indexes()

if not msra_character_dict:
    msra_character_dict = load_msra_character_indexes()

if not phrase_dict:
    phrase_dict = load_phrase_indexes()

if not msra_phrase_dict:
    msra_phrase_dict = load_msra_phrase_indexes()

if not label_dict:
    label_dict = load_label_indexes(dataset_type=Dataset.WEIBO)

if not msra_label_dict:
    msra_label_dict = load_label_indexes(dataset_type=Dataset.MSRA)

if not char2radical:
    with open('process_character/data_preprocess/char2radical.txt', 'r', encoding='utf-8') as f:
        for line in f:
            char, radical = line.split()
            char2radical[char] = radical_dict[radical]
