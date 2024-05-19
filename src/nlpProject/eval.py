import torch
import numpy as np
from pathlib import Path
import pickle
import time
import os

from nlpProject.make_data import DataMaker
from nlpProject.rnn_baseline import RNN
from nlpProject.one_layer_lstm import LSTM1
from nlpProject.utils import get_metrics_n, get_bleu, compute_loss

def make_val(data_maker, val_text, seq_len):
    data = data_maker.encode_string(val_text).t()
    X, Y = data[:-1, :], data[1:, :]
    n_chars, n_features = X.shape
    n_samples = n_chars // seq_len
    X_trunc, Y_trunc = X[:n_samples * seq_len], Y[:n_samples * seq_len]
    X_resh, Y_resh = X_trunc.view(n_samples, seq_len, n_features).permute(2, 1, 0), Y_trunc.view(n_samples, seq_len, n_features).permute(2, 1, 0)
    return X_resh, Y_resh

def synth_from_val(model, data_maker, X_val, Y_val, length = 1000, T = 1.0):
    indexes = []
    P = []
    h_0, c_0 = (torch.zeros(model.hidden_size, 1, dtype=torch.double),
                torch.zeros(model.hidden_size, 1, dtype=torch.double))
    with torch.no_grad():
        X_in = X_val[:, :, 0]
        p_init, (h_t, c_t) = model(X_in.view(-1, X_val.shape[1], 1), (h_0, c_0), T)
        P.append(p_init)
        t = 0
        CP = torch.cumsum(p_init[:,-1,:], dim=0)
        a = torch.rand(1)
        ixs = torch.where(CP - a > 0)
        ii = ixs[0][0].item()
        xt = torch.zeros((X_in.size(0), 1, 1), dtype=torch.double)
        xt[ii, 0, 0] = 1
        x_input = xt
        while t < (length - X_in.shape[1] - 1):
            p, (h_t, c_t) = model(x_input, (h_t, c_t), T)
            P.append(p)
            CP = torch.cumsum(p, dim=0)
            a = torch.rand(1)
            ixs = torch.where(CP - a > 0)
            ii = ixs[0][0].item()
            indexes.append(ii)
            xt = torch.zeros((x_input.size(0), 1, 1), dtype=torch.double)
            xt[ii, 0, 0] = 1
            x_input = xt
            t += 1
        print(len(indexes))
        print(len(P))
        Y = []
        for idx in indexes:
            oh = [0]*x_input.size(0)
            oh[idx] = 1
            Y.append(oh)
        Y = torch.tensor(Y).t()
        Y = torch.cat((Y_val[:, :, 0], Y), dim=1)
        P = torch.cat(tuple(P), dim=1)
        s = ''
        for i in range(Y.shape[1]):
            idx = torch.where(Y[:, i] == 1)[0].item()
            s += data_maker.ind_to_char[idx]
    return P, Y, s

def eval_lstm1(model_path, data_path, validation_path, seq_len, T = 1.0):
    state_dict = torch.load(Path(model_path))
    input_size, hidden_size = state_dict["Wf"].size()[1], state_dict["Wf"].size()[0]
    model = LSTM1(input_size, hidden_size, input_size)
    model.load_state_dict(state_dict)
    model.eval()
    data_maker = DataMaker(data_path)
    with open(validation_path, 'r') as f:
        text = f.read()
    X_val, Y_val = make_val(data_maker, text, seq_len)
    P, Y, s = synth_from_val(model, data_maker, X_val, Y_val, length=len(text), T=T)
    prediction_loss = compute_loss(Y.view_as(P), P).item()
    generated_fpath = f'./reports/logs/generated_val_{time.time()}.txt'
    with open(generated_fpath, 'w+') as f:
        f.write(text[0] + s)
    print(f"------------------------------------------------")
    print(f"Prediction loss: {prediction_loss:.4f}")
    prec, rec, fm = get_metrics_n(validation_path, generated_fpath, 1)
    bleu = get_bleu(validation_path, generated_fpath)
    print(f'Precision: {prec:.2f}% - Recall: {rec:.2f}% - F-measure: {fm:.2f}% - BLEU: {bleu:.2f}%')
    print(f"------------------------------------------------")
    return prediction_loss, prec, rec, fm, bleu