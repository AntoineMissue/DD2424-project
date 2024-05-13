import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
import time
import os

from nlpProject.make_data import DataMaker
from nlpProject.rnn_baseline import RNN
from nlpProject.utils import get_metrics_n, get_bleu

def rnn_generate(rnn_filename, first_char = ' ', length = 10000, T = 1.0):
    with open(Path(f'./models/RNN/{rnn_filename}'), 'rb') as handle:
        rnn_dict = pickle.load(handle)
        data_maker = DataMaker(rnn_dict['book_fname'])
    test_rnn = RNN(hidden_size=rnn_dict['hidden_size'], seq_length=rnn_dict['seq_length'], data_path=rnn_dict['book_fname'])
    test_rnn.params = {
        'W': rnn_dict['W'], 
        'U': rnn_dict['U'],
        'V': rnn_dict['V'],
        'b': rnn_dict['b'],
        'c': rnn_dict['c']
    }
    x_input = data_maker.encode_string(first_char)
    _, s_t = test_rnn.synthetize_seq(
        torch.zeros((test_rnn.hidden_size, 1), dtype=torch.double), 
        x_input[:,0], length, T)
    sequence = first_char + s_t
    return sequence

if __name__ == '__main__':
    print(rnn_generate('rnn_adagrad_test.pickle'))