import numpy as np
import torch
from pathlib import Path

class DataMaker:
    def __init__(self, book_fname = './data/shakespeare.txt'):
        self.book_fname = book_fname
        self.book_data, self.input_size, self.char_to_ind, self.ind_to_char = self.make_charmap()

    def make_charmap(self):
        book_path = Path(self.book_fname)
        with open(book_path, 'r') as book:
            book_data = book.read()

        word_list = book_data.split()
        chars = [[*word] for word in word_list]
        max_len = max(len(word) for word in chars)
        for wordl in chars:
            while len(wordl) < max_len:
                wordl.append(' ')
        chars = np.array(chars)

        unique_chars = list(np.unique(chars))
        unique_chars.append('\n')
        unique_chars.append('\t')

        char_to_ind = {}
        ind_to_char = {}
        for idx, char in enumerate(unique_chars):
            char_to_ind[char] = idx
            ind_to_char[idx] = char

        return book_data, len(unique_chars), char_to_ind, ind_to_char

    def encode_char(self, char):
        oh = [0]*self.input_size
        oh[self.char_to_ind[char]] = 1
        return oh

    def encode_string(self, chars):
        M = []
        for i in range(len(chars)):
            M.append(self.encode_char(chars[i]))
        M = torch.tensor(M, dtype=torch.double).t()
        return M

    def split_text(self, text, train_frac=0.8, val_frac=0.1):
        train_end = int(len(text) * train_frac)
        val_end = train_end + int(len(text) * val_frac)

        train_data = text[:train_end]
        val_data = text[train_end:val_end]
        test_data = text[val_end:]

        return train_data, val_data, test_data
