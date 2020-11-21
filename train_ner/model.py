from configparser import ConfigParser

from numpy.random import seed

from keras_contrib.layers import CRF
from tensorflow import set_random_seed

from keras import Input
from keras.models import Model
from keras.layers import Bidirectional, Conv1D, concatenate, Dropout, Embedding, GRU, LSTM
import numpy as np

from process_character.constants import Dataset
from train_status import TrainConf
from train_ner import process_data

cf = ConfigParser()
cf.read(r'train_ner/train.cfg')
EMBED_DIM = int(cf.get('size', 'EMBED_DIM'))
CHAR_AMOUNT = int(cf.get('size', 'CHAR_AMOUNT'))
MSRA_CHAR_AMOUNT = int(cf.get('msra_size', 'CHAR_AMOUNT'))
WORD_AMOUNT = int(cf.get('size', 'WORD_AMOUNT'))
MSRA_WORD_AMOUNT = int(cf.get('msra_size', 'WORD_AMOUNT'))
RAD_AMOUNT = int(cf.get('size', 'RAD_AMOUNT'))
MSRA_RAD_AMOUNT = int(cf.get('msra_size', 'RAD_AMOUNT'))
TIME_STEPS = int(cf.get('size', 'TIME_STEPS'))
BATCH_SIZE = int(cf.get('hyperparameter', 'BATCH_SIZE'))
EPOCH = int(cf.get('hyperparameter', 'EPOCH'))
CONV_FILTERS = int(cf.get('hyperparameter', 'CONV_FILTERS'))
GRU_SIZE = int(cf.get('hyperparameter', 'GRU_SIZE'))
LSTM_SIZE = int(cf.get('hyperparameter', 'LSTM_SIZE'))
CONV_KERNEL_SIZE = int(cf.get('hyperparameter', 'CONV_KERNEL_SIZE'))


class MyModel:
    def __init__(
            self,
            word_embedding_matrix,
            char_embedding_matrix,
            conf=None,
            seed=None,
    ):
        self.model = None
        self.word_embedding_matrix = word_embedding_matrix
        self.char_embedding_matrix = char_embedding_matrix
        self.conf = conf
        self.seed = seed
        (
            self.train_x_c,
            self.train_x_w,
            self.train_x_rad,
            self.train_y,
        ), (
            self.test_x_c,
            self.test_x_w,
            self.test_x_rad,
            self.test_y,
        ), (
            self.dev_x_c,
            self.dev_x_w,
            self.dev_x_rad,
            self.dev_y,
        ) = process_data.load_data(dataset=self.conf.get('dataset'))

    def train(self):
        is_weibo = self.conf.get('dataset') == Dataset.WEIBO
        char_input = Input(shape=self.train_x_c.shape[1:], name='char_input')
        word_input = Input(shape=self.train_x_w.shape[1:], name='word_input')
        rad_input = Input(shape=self.train_x_rad.shape[1:], name='rad')

        char_embedding = Embedding(
            input_dim=CHAR_AMOUNT if is_weibo else MSRA_CHAR_AMOUNT,
            weights=[self.char_embedding_matrix],
            output_dim=EMBED_DIM,
            trainable=True,
        )(char_input)
        word_embedding = Embedding(
            input_dim=WORD_AMOUNT if is_weibo else MSRA_WORD_AMOUNT,
            weights=[self.word_embedding_matrix],
            output_dim=EMBED_DIM,
            trainable=True,
        )(word_input)

        input_list = [char_input, word_input]
        concat_list = [word_embedding]
        train_dict = {
            'char_input': self.train_x_c,
            'word_input': self.train_x_w,
        }
        dev_dict = {
            'char_input': self.dev_x_c,
            'word_input': self.dev_x_w,
        }
        if self.conf.get('with_radical'):
            rad_embedding = Embedding(input_dim=RAD_AMOUNT if is_weibo else MSRA_RAD_AMOUNT,
                                      output_dim=EMBED_DIM)(rad_input)
            conv_rad = Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same')(
                Dropout(0.2)(rad_embedding))
            concat_list.append(conv_rad)
            input_list.append(rad_input)
            train_dict['rad'] = self.train_x_rad
            dev_dict['rad'] = self.dev_x_rad

        if self.conf.get('network') == TrainConf.cnn:
            conv_char = Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same')(
                Dropout(0.2)(char_embedding))
            concat_list.append(conv_char)

        if self.conf.get('network') == TrainConf.bilstm:
            bilstm_char = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout=0.5))(
                Dropout(0.2)(char_embedding))
            concat_list.append(bilstm_char)

        if self.conf.get('network') == TrainConf.conv_gru:
            bigru_char = Bidirectional(GRU(GRU_SIZE, dropout=0.5, return_sequences=True))(
                Dropout(0.2)(
                    char_embedding))
            convgru_char = Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE,
                                  padding='same')(bigru_char)
            concat_list.append(bigru_char)
            concat_list.append(convgru_char)

        concat = concatenate(concat_list)

        if self.conf.get('tagger') == TrainConf.bigru_crf:
            bigru_or_lstm = Bidirectional(GRU(GRU_SIZE, dropout=0.5, return_sequences=True))(
                Dropout(0.2)(concat))

        if self.conf.get('tagger') == TrainConf.bilstm_crf:
            bigru_or_lstm = Bidirectional(LSTM(LSTM_SIZE, dropout=0.5, return_sequences=True))(
                Dropout(0.2)(concat))

        crf = CRF(CHUCK_TAGS_NUM, sparse_target=False)
        output = crf(Dropout(0.2)(bigru_or_lstm))
        self.model = Model(inputs=input_list,
                           outputs=output)
        if self.seed:
            seed(self.seed)
            set_random_seed(self.seed)
        self.model.compile(optimizer='adam', loss=crf.loss_function,
                           metrics=[crf.accuracy])
        self.model.summary()
        if self.seed:
            seed(self.seed)
            set_random_seed(self.seed)
        self.model.fit(train_dict,
                       self.train_y,
                       validation_data=[dev_dict, self.dev_y],
                       batch_size=BATCH_SIZE,
                       epochs=EPOCH,
                       )

    def predict(self, test_list):
        if self.seed:
            seed(self.seed)
            set_random_seed(self.seed)
        return self.model.predict(test_list)


def calculate_precision_recall_f1(predict, ground_truth, matrix_shape):
    predict = np.argmax(predict, predict.ndim - 1)
    predict = predict.reshape(-1)
    ground_truth = np.argmax(ground_truth, ground_truth.ndim - 1)
    ground_truth = ground_truth.reshape(-1)

    print('predict shape is ', predict.shape)
    print('ground_truth shape is ', ground_truth.shape)

    for i in [1, 3, 5, 7]:
        print("predict %s: %s times" % (i, np.sum(predict == i)))
    for i in [1, 3, 5, 7]:
        print("ground_truth %s: %s times" % (i, np.sum(ground_truth == i)))

    precision_denominator = 0
    recall_denominator = 0
    numerator = 0
    confusion_matrix = np.zeros(dtype='int', shape=matrix_shape)

    match_candidate = (0, 0)
    for predict_label, gt_label in zip(predict, ground_truth):
        if predict_label in [1, 3, 5, 7]:  # B-XXX
            precision_denominator += 1
            pred_category = (predict_label + 1) // 2  # 0-NIL, 1-ORG, 2-LOC, 3-PER, 4-GPE, 5-O
            gt_category = (gt_label + 1) // 2
            confusion_matrix[pred_category][gt_category] += 1

        if gt_label in [1, 3, 5, 7]:  # B-XXX
            recall_denominator += 1
            pred_category = (predict_label + 1) // 2  # 0-NIL, 1-ORG, 2-LOC, 3-PER, 4-GPE, 5-O
            gt_category = (gt_label + 1) // 2
            confusion_matrix[pred_category][gt_category] += 1

        # O, NIL or B-XXX, means the previous NER complete
        if gt_label in [0, 1, 3, 5, 7, 9] and \
                predict_label in [0, 1, 3, 5, 7, 9] and (
                match_candidate[0] > 0 or match_candidate[1] > 0):
            match_candidate = (0, 0)
            numerator += 1
        if predict_label == gt_label and predict_label in [1, 3, 5, 7]:
            match_candidate = (predict_label, gt_label)
        if predict_label != gt_label:
            match_candidate = (0, 0)

    precision = numerator / precision_denominator
    recall = numerator / recall_denominator
    f1 = 2 * precision * recall / (precision + recall)
    print(precision, recall, f1)
    return precision, recall, f1, confusion_matrix


def run(conf=None):
    dataset = conf.get('dataset')
    seeds = range(5)
    embedding_data = []
    with open('process_character/data_preprocess/{}embedding_data.txt'.format(
            '' if dataset == Dataset.WEIBO else 'msra_'
    ), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            embedding_data.append(line.split(' ')[1:])
    embedding_data.insert(0, [0] * EMBED_DIM)
    embedding_data = np.array(embedding_data)  # 11888, 200

    char_embedding_data = []
    with open('process_character/data_preprocess/{}char_embedding_data.txt'.format(
            '' if dataset == Dataset.WEIBO else 'msra_'
    ), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            char_embedding_data.append(line.split(' ')[1:])
    char_embedding_data.insert(0, [0] * EMBED_DIM)
    char_embedding_data = np.array(char_embedding_data)  # 3191, 200

    ps = []
    rs = []
    fs = []
    for i in range(len(seeds)):
        seed = seeds[i]
        print('Running {}/{}:'.format(i, len(seeds)))
        my_model = MyModel(
            word_embedding_matrix=embedding_data,
            char_embedding_matrix=char_embedding_data,
            conf=conf,
            seed=seed,
        )
        my_model.train()
        test_list = [my_model.test_x_c, my_model.test_x_w]
        if my_model.conf.get('with_radical') == TrainConf.with_radical:
            test_list.append(my_model.test_x_rad)
        predict = my_model.predict(test_list)

        np.save('train_ner/predict', predict)
        np.save('train_ner/ground_truth', my_model.test_y)
        predict = np.load('train_ner/predict.npy')
        gt = np.load('train_ner/ground_truth.npy')

        p, r, f, _ = calculate_precision_recall_f1(
            predict, gt, (6, 6) if dataset == Dataset.WEIBO else (28, 28)
        )
        ps.append(p)
        rs.append(r)
        fs.append(f)

    print('precisions are : {}'.format(ps))
    print('recalls are : {}'.format(rs))
    print('F1s are : {}'.format(fs))

    print('average output:')
    result_str1 = 'precision = %s' % (sum(ps) / len(ps))
    result_str2 = 'recall = %s' % (sum(rs) / len(ps))
    result_str3 = 'f1 = %s' % (sum(fs) / len(ps))
    print(result_str1)
    print(result_str2)
    print(result_str3)
    with open('train_ner/data.txt', 'w', encoding='utf-8') as f:
        f.write(result_str1 + '\n')
        f.write(result_str2 + '\n')
        f.write(result_str3 + '\n')


if __name__ == '__main__':
    run(conf={
        'dataset': Dataset.MSRA,
        'with_radical': int(TrainConf.with_radical),
        'network': TrainConf.conv_gru,
        'tagger': TrainConf.bigru_crf,
    })
