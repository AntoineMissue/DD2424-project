import torch
import numpy as np
from pathlib import Path
import pickle
import time
import os

from nlpProject.make_data import DataMaker
from nlpProject.rnn_baseline import RNN
from nlpProject.utils import get_metrics_n, get_bleu

def synthesize_seq_lstm1(model, h_t, c_t, x0, length = 1000):
    indexes = []
    with torch.no_grad():
        data_maker = DataMaker()
        x_input = x0.reshape(-1, 1, 1)
        t = 0
        while t < length:
            P, (h_t, c_t) = model(x_input, (h_t, c_t))
            CP = torch.cumsum(P, dim=0)
            a = torch.rand(1)
            ixs = torch.where(CP - a > 0)
            ii = ixs[0][0].item()
            indexes.append(ii)
            xt = torch.zeros((x_input.size(0), 1, 1), dtype=torch.double)
            xt[ii, 0, 0] = 1
            x_input = xt
            t += 1
        Y = []
        for idx in indexes:
            oh = [0]*x_input.size(0)
            oh[idx] = 1
            Y.append(oh)
        Y = torch.tensor(Y).t()
        s = ''
        for i in range(Y.shape[1]):
            idx = torch.where(Y[:, i] == 1)[0].item()
            s += data_maker.ind_to_char[idx]
    return Y, s

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

def rnn_metrics(rnn_filename, book_path, grams = 1, length = 10000, T = 1.0, clean = True):
    text = rnn_generate(rnn_filename, length = length, T = T)
    generated_fpath = f'./reports/logs/generated_text_{time.time()}.txt'
    with open(generated_fpath, 'w+') as text_file:
        text_file.write(text)
    prec, rec, fm = get_metrics_n(book_path, generated_fpath, grams)
    bleu = get_bleu(book_path, generated_fpath)
    if clean:
        os.remove(generated_fpath)
    return prec, rec, fm, bleu

def rnn_evaluation(rnn_filename, book_path, tries = 10, length = 10000, T = 1.0, clean = True):
    precs, recs, fms, bleus = [], [], [], []
    for i in range(tries):
        current_prec, current_rec, current_fm, current_bleu = rnn_metrics(rnn_filename, book_path, grams = 1, length = length, T = T, clean = clean)
        precs.append(current_prec)
        recs.append(current_rec)
        fms.append(current_fm)
        bleus.append(current_bleu)
        print(f'Try {i+1} - Precision: {current_prec:.2f}% - Recall: {current_rec:.2f}% - F-measure: {current_fm:.2f}% - BLEU: {current_bleu:.2f}%')
    precs, recs, fms, bleus = np.array(precs), np.array(recs), np.array(fms), np.array(bleus)
    mean_prec, mean_rec, mean_fm, mean_bleu, std_prec, std_rec, std_fm, std_bleu = precs.mean(), recs.mean(), fms.mean(), bleus.mean(), precs.std(), recs.std(), fms.std(), bleus.std()
    return {'precision': precs, 'recall': recs, 'fmeasure': fms, 'bleu': bleus}, {'precision': mean_prec, 'recall': mean_rec, 'fmeasure': mean_fm, 'bleu': mean_bleu}, {'precision': std_prec, 'recall': std_rec, 'fmeasure': std_fm, 'bleu': std_bleu}
    
if __name__ == '__main__':
    pcts, mean, std = rnn_evaluation('rnn_adagrad_test.pickle', './data/shakespeare.txt')
    
    print(f"Precision - Mean: {mean['precision']:.2f}% ; Standard deviation: {std['precision']:.2f}%")
    print(f"Recall - Mean: {mean['recall']:.2f}% ; Standard deviation: {std['recall']:.2f}%")
    print(f"F-measure - Mean: {mean['fmeasure']:.2f}% ; Standard deviation: {std['fmeasure']:.2f}%")
    print(f"BLEU - Mean: {mean['bleu']:.2f}% ; Standard deviation: {std['bleu']:.2f}%")