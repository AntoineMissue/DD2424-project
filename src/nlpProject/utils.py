from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

def make_dataloader(data_maker, seq_len, batch_size, shuffle = False):
    data = data_maker.encode_string(data_maker.book_data).t()
    X, Y = data[:-1, :], data[1:, :]
    n_chars, n_features = X.shape
    n_samples = n_chars // seq_len
    X_trunc, Y_trunc = X[:n_samples * seq_len], Y[:n_samples * seq_len]
    X_resh, Y_resh = X_trunc.view(n_samples, seq_len, n_features), Y_trunc.view(n_samples, seq_len, n_features)
    dataset = TensorDataset(X_resh, Y_resh)
    dataloader = DataLoader(dataset, batch_size, shuffle, drop_last=True)
    return dataloader

def compute_loss(Y, P):
    batch_size = Y.shape[2]
    log_probs = torch.log(P)
    cross_entropy = -torch.sum(Y * log_probs)
    loss = cross_entropy / batch_size
    return loss

def compute_loss_lstm2(Y, P):
    batch_size = Y.shape[0]
    log_probs = torch.log(P)
    cross_entropy = -torch.sum(Y * log_probs)
    loss = cross_entropy / batch_size
    return loss

def process_word(word):
    word = word.replace('.', '')
    word = word.replace(':', '')
    word = word.replace(',', '')
    word = word.replace('?', '')
    word = word.replace('!', '')
    word = word.replace(';', '')
    word = word.lower()
    return word

def get_ngrams(n, text_path):
    path = Path(text_path)
    with open(path, 'r') as text_f:
        text = text_f.read()
    
    split_text = text.split()
    processed = [process_word(split_text[idx]) for idx in range(len(split_text))]

    ngrams = []
    for i in range(len(processed) - n + 1):
        seq = ''
        for j in range(i, i + n):
            seq += processed[j]
            if j != (i + n - 1):
                seq += ' '
        ngrams.append(seq)

    return ngrams

def get_metrics_n(book_path, output_path, n):
    book_ngrams = get_ngrams(n, book_path)
    output_ngrams = get_ngrams(n, output_path)
    correct = 0
    for gram in output_ngrams:
        if gram in book_ngrams:
            correct += 1
    precision = (correct / len(output_ngrams))*100
    recall = (correct / len(book_ngrams))*100
    if precision + recall > 1e-8:
        f_measure = (precision * recall)/((precision + recall)/2)
    else:
        f_measure = 0
    return precision, recall, f_measure

def get_bleu(book_path, output_path):
    book_1grams = get_ngrams(1, book_path)
    output_1grams = get_ngrams(1, output_path)
    reference_length = len(book_1grams)
    output_length = len(output_1grams)
    fact = min(1, output_length / reference_length)
    precs = torch.tensor([get_metrics_n(book_path, output_path, i)[0] for i in range(1, 5)])
    return (fact * torch.pow(torch.prod(precs), 1/4)).item()

