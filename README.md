# ME-CNER
Code for CIKM 2019 paper ["Exploiting Multiple Embeddings for Chinese Named Entity Recognition"](https://arxiv.org/abs/1908.10657).

## Citation
If you use this code in your work, please kindly cite our work:
```bibtex
@inproceedings{cikm19:xu,
  author    = {Canwen Xu and
               Feiyang Wang and
               Jialong Han and
               Chenliang Li},
  title     = {Exploiting Multiple Embeddings for Chinese Named Entity Recognition},
  booktitle = {The 28th ACM International Conference on Information and Knowledge Management, {CIKM} 2019, Beijing, China,
               November 3-7, 2019},
  publisher = {{ACM}},
  year      = {2019},
  url       = {https://doi.org/10.1145/3357384.3358117},
  doi       = {10.1145/3357384.3358117}
}
```

## Requirement
	Python: 3.6  
	Keras: 2.2.2
	Keras-contrib: 2.0.8
	jieba: 0.39
	

## Dataset
We use a standard Weibo NER dataset provided by [Peng and Dredze, 2015](http://aclweb.org/anthology/D/D15/D15-1064.pdf),
and a formal MSRA News dataset provided by [Levow, 2006](https://www.aclweb.org/anthology/W06-0115).

## Pretrained Embeddings
The pretrained character and word embeddings are provided by [Tencent AI Lab](https://ai.tencent.com/ailab/nlp/en/embedding.html). Download it [here](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz).

The radical embedding is randomly initialized.

## How to Run
1. Install all requirements
```shell
pip install keras==2.2.2  # for Keras
pip install git+https://www.github.com/keras-team/keras-contrib.git  # for CRF layer
pip install jieba  # for word segmentation 
```

2. Download pretrained embeddings
Download [Tencent Embeddings](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz), extract it and put it in `process_data/data_preprocess`.

3. Run the pre-processing code
```shell
python concat_data.py
```

4. Run the model (with different config)
```shell
python main.py --dataset ${weibo/msra} --with_radical ${1/0} --network ${convgru/cnn/bilstm} 
--tagger ${bigrucrf/bilstmcrf} --entity_type ${all/nm/ne}
```
```
dataset:
  weibo
  msra
  
with_radical:  # input radical embedding or not
  0  # no radical embedding input, only word embedding and char embedding
  1  # with radical embedding
  
network:  # for characters
  convgru  # Conv-GRU  
  bilstm 
  cnn 
  
tagger:
  bigrucrf  # Bidirectional GRU-CRF 
  bilstmcrf  # Bidirectional LSTM-CRF 
  
entity_type:
  ne  # only Named Entity. e.g. 王小明 (Xiaoming Wang), 北京市 (Beijing City)
  nm  # only Nominal Mention. e.g. 班长 (class president), 妈妈 (mother) 
  all  # take both Named Entity and Nominal Mention into accounts
```

For example, run the following shell to run our final ME-CNER model on WEIBO dataset, but only recognize named 
entities (all nominal mentions are ignored).
```shell
python main.py --dataset weibo --with_radical 1 --network convgru --tagger bigrucrf --entity_type ne
```

