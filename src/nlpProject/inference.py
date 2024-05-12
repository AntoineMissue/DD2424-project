import pickle
import torch
from pathlib import Path
from nlpProject.make_data import DataMaker
from nlpProject.rnn_baseline import RNN

def rnn_generate(rnn_filename, book_fname = './data/shakespeare.txt', hidden_size = 100, seq_length = 25, first_char = ' ', length = 1000, T = 1.0):
    data_maker = DataMaker(book_fname)
    with open(Path(f'./models/RNN/{rnn_filename}'), 'rb') as handle:
        params = pickle.load(handle)
    test_rnn = RNN(hidden_size=hidden_size, seq_length=seq_length, data_path=book_fname)
    test_rnn.params = params
    x_input = data_maker.encode_string(first_char)
    _, s_t = test_rnn.synthetize_seq(
        torch.zeros((test_rnn.hidden_size, 1), dtype=torch.double), 
        x_input[:,0], length, T)
    sequence = first_char + s_t
    return sequence

if __name__ == '__main__':
    print(rnn_generate('rnn_adagrad_test.pickle', length=10000))