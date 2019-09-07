# -*- coding: utf-8 -*-
import argparse

from entity_type import EntityType
from process_character.load_embedding_data import run as run_load_embedding_data
from process_character.load_weibo_data import run as run_load_weibo_data
from train_ner.model import run as run_train
from train_status import TrainConf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=TrainConf.weibo)
    parser.add_argument('--with_radical', default=TrainConf.with_radical)
    parser.add_argument('--network', default=TrainConf.conv_gru)
    parser.add_argument('--tagger', default=TrainConf.bigru_crf)
    parser.add_argument('--entity_type', default=EntityType.all)
    args = parser.parse_args()
    entity_type = args.entity_type

    run_load_weibo_data(entity_type=entity_type)
    run_load_embedding_data()
    run_train(conf={
        'dataset': args.dataset,
        'with_radical': int(args.with_radical),
        'network': args.network,
        'tagger': args.tagger,
    })
