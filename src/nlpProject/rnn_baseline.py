import pickle
import time
import torch
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from nlpProject.make_data import DataMaker
from nlpProject.utils import compute_loss, make_dataloader
from nlpProject import logger

class RNN:
    def __init__(self, hidden_size = 100, learning_rate = 0.1, epsilon = 1e-8, seq_length = 25, data_path = "./data/shakespeare.txt"):
        ## Constants
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size  # dimensionality of the hidden state
        self.learning_rate = learning_rate  # learning rate
        self.epsilon = epsilon  # for AdaGrad/Adam
        self.seq_length = seq_length  # length of input sequences used during training
        self.book_fname = data_path
        self.data_maker = DataMaker(self.book_fname)
        self.book_data, self.input_size, self.char_to_ind, self.ind_to_char = self.data_maker.make_charmap()

        ## Parameters
        # bias vectors
        self.b = torch.zeros((self.hidden_size, 1), dtype=torch.double, device=self.device)
        self.c = torch.zeros((self.input_size, 1), dtype=torch.double, device=self.device)
        # weight matrices
        sig = 2/(self.hidden_size + self.input_size)
        self.U = torch.normal(0.0, sig, (self.hidden_size, self.input_size), dtype=torch.double, device=self.device)
        self.W = torch.normal(0.0, sig, (self.hidden_size, self.hidden_size), dtype=torch.double, device=self.device)
        self.V = torch.normal(0.0, sig, (self.input_size, self.hidden_size), dtype=torch.double, device=self.device)
        self.h0 = torch.zeros((self.hidden_size, 1), dtype=torch.double, device=self.device)
        self.params = {
            'W': self.W, 
            'U': self.U,
            'V': self.V,
            'b': self.b,
            'c': self.c
        }
        logger.info("Model initialized")

    def forward(self, X, hprev):
        _, _, batch_size = X.shape

        P = torch.zeros((self.input_size, self.seq_length, batch_size), dtype=torch.double)
        A = torch.zeros((self.hidden_size, self.seq_length, batch_size), dtype=torch.double)
        H = torch.zeros((self.hidden_size, self.seq_length, batch_size), dtype=torch.double)

        ht = hprev.clone()
        for i in range(self.seq_length):
            xt = X[:, i, :]
            at = torch.mm(self.params['W'], ht) + torch.mm(self.params['U'], xt) + self.params['b'].expand(self.hidden_size, batch_size)
            ht = torch.tanh(at)
            ot = torch.mm(self.params['V'], ht) + self.params['c'].expand(self.input_size, batch_size)
            pt = F.softmax(ot, dim=0)

            H[:, i, :] = ht
            P[:, i, :] = pt
            A[:, i, :] = at

        return A, H, P, ht

    def backward(self, X, Y, A, H, P, hprev):
        dA = torch.zeros_like(A)
        dH = torch.zeros_like(H)

        G = -(Y - P)
        dV = torch.bmm(G.permute(2, 0, 1), H.permute(2, 1, 0)).mean(dim=0)
        dhtau = torch.matmul(G[:, -1, :].t(), self.params['V']).t()
        datau = (1 - torch.pow(torch.tanh(A[:, -1, :]), 2)) * dhtau
        dH[:, -1, :] = dhtau
        dA[:, -1, :] = datau

        for i in range(self.seq_length - 2, -1, -1):
            dht = torch.matmul(G[:, i, :].t(), self.params['V']).t() + torch.matmul(dA[:, i+1, :].t(), self.params['W']).t()
            dat = (1 - torch.pow(torch.tanh(A[:, i]), 2)) * dht
            dH[:, i] = dht
            dA[:, i] = dat

        Hd = torch.cat((hprev.reshape((self.hidden_size, 1, -1)), H[:, :-1, :]), dim=1)
        dW = torch.matmul(dA.permute(2, 0, 1), Hd.permute(2, 1, 0)).mean(dim=0)
        dU = torch.matmul(dA.permute(2, 0, 1), X.permute(2, 1, 0)).mean(dim=0)
        dc = G.sum(1).mean(dim=1).reshape((-1, 1))
        db = dA.sum(1).mean(dim=1).reshape((-1, 1))
        grads = {'U': dU, 'W': dW, 'V': dV, 'c': dc, 'b': db}
        grads_clamped = {k: torch.clamp(v, min=-5.0, max=5.0) for (k,v) in grads.items()}
        return grads, grads_clamped

    def synthetize_seq(self, h0, x0, n, T = 1):
        t, ht, xt = 0, h0, x0
        indexes = []
        while t < n:
            xt = xt.reshape((self.input_size, 1))
            at = torch.mm(self.params['W'], ht) + torch.mm(self.params['U'], xt) + self.params['b']
            ht = torch.tanh(at)
            ot = torch.mm(self.params['V'], ht) + self.params['c']
            pt = F.softmax(ot/T, dim=0)
            cp = torch.cumsum(pt, dim=0)
            a = torch.rand(1)
            ixs = torch.where(cp - a > 0)
            ii = ixs[0][0].item()
            indexes.append(ii)
            xt = torch.zeros((self.input_size, 1), dtype=torch.double)
            xt[ii, 0] = 1
            t += 1
        Y = []
        for idx in indexes:
            oh = [0]*self.input_size
            oh[idx] = 1
            Y.append(oh)
        Y = torch.tensor(Y).t()
        s = ''
        for i in range(Y.shape[1]):
            idx = torch.where(Y[:, i] == 1)[0].item()
            s += self.ind_to_char[idx]
        return Y, s
    
    def train_adagrad(self, batch_size, n_epochs, synth_interval = 5, T = 1, save = False):
        logger.info("Start of AdaGrad training.")
        step = 0
        smooth_loss = 0
        losses = []
        dataloader = make_dataloader(self.data_maker, self.seq_length, batch_size)

        mb = torch.zeros_like(self.params['b'], dtype=torch.double)
        mc = torch.zeros_like(self.params['c'], dtype=torch.double)
        mU = torch.zeros_like(self.params['U'], dtype=torch.double)
        mV = torch.zeros_like(self.params['V'], dtype=torch.double)
        mW = torch.zeros_like(self.params['W'], dtype=torch.double)
        ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs} - Step {step + 1}")
            hprev = torch.zeros((self.hidden_size, batch_size), dtype=torch.double)

            for X_train, Y_train in tqdm.tqdm(dataloader, desc="Processing batches"):
                X_train = X_train.permute(2, 1, 0)
                Y_train = Y_train.permute(2, 1, 0)

                A_train, H_train, P_train, hts = self.forward(X_train, hprev)
                loss = compute_loss(Y_train, P_train)
                grads, grads_clamped = self.backward(X_train, Y_train, A_train, H_train, P_train, hprev)

                for k in ms.keys():
                    ms[k] += grads_clamped[k]**2
                    self.params[k] -= (self.learning_rate/torch.sqrt(ms[k] + self.epsilon)) * grads_clamped[k]

                if step == 0:
                    smooth_loss = loss.item()
                else:
                    smooth_loss = 0.999*smooth_loss + 0.001*loss.item()
                losses.append(smooth_loss)

                step += 1
                hprev = hts

            print(f"\t * Smooth loss: {smooth_loss:.4f}")
            if (epoch + 1) % synth_interval == 0:
                _, s_syn = self.synthetize_seq(hprev[:, 0:1], X_train[:, 0, 0], 200, T)
                print("-" * 100)
                print(f"Synthesized sequence: \n{s_syn}")
                print("-" * 100)
                
        if save:
            with open(Path(f'./models/RNN/rnn_adagrad_{time.time()}.pickle'), 'wb') as handle:
                pickle.dump({
                    **self.params, 
                    'book_fname': self.book_fname, 
                    'hidden_size': self.hidden_size, 
                    'learning_rate': self.learning_rate, 
                    'epsilon': self.epsilon, 
                    'seq_length': self.seq_length
                    }, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("End of AdaGrad training.")

        return losses

    def train_adam(self, batch_size, n_epochs, synth_interval = 5, beta_1 = 0.9, beta_2 = 0.999, T = 1, save = False):
        logger.info("Start of Adam training.")
        step = 0
        smooth_loss = 0
        losses = []
        dataloader = make_dataloader(self.data_maker, self.seq_length, batch_size)

        mb = torch.zeros_like(self.params['b'], dtype=torch.float)
        vb = torch.zeros_like(self.params['b'], dtype=torch.float)
        mc = torch.zeros_like(self.params['c'], dtype=torch.float)
        vc = torch.zeros_like(self.params['c'], dtype=torch.float)
        mU = torch.zeros_like(self.params['U'], dtype=torch.float)
        vU = torch.zeros_like(self.params['U'], dtype=torch.float)
        mV = torch.zeros_like(self.params['V'], dtype=torch.float)
        vV = torch.zeros_like(self.params['V'], dtype=torch.float)
        mW = torch.zeros_like(self.params['W'], dtype=torch.float)
        vW = torch.zeros_like(self.params['W'], dtype=torch.float)
        ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}
        vs = {'b': vb, 'c': vc, 'U': vU, 'V': vV, 'W': vW}

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs} - Step {step + 1}")
            hprev = torch.zeros((self.hidden_size, batch_size), dtype=torch.double)

            for X_train, Y_train in tqdm.tqdm(dataloader, desc="Processing batches"):
                X_train = X_train.permute(2, 1, 0)
                Y_train = Y_train.permute(2, 1, 0)

                A_train, H_train, P_train, hts = self.forward(X_train, hprev)
                loss = compute_loss(Y_train, P_train)
                grads, grads_clamped = self.backward(X_train, Y_train, A_train, H_train, P_train, hprev)

                for k in ms.keys():
                    ms[k] = beta_1*ms[k] + (1 - beta_1)*grads_clamped[k]
                    vs[k] = beta_2*vs[k] + (1 - beta_2)*(grads_clamped[k]**2)
                    m_hat = ms[k]/(1 - beta_1**(step+1))
                    v_hat = vs[k]/(1 - beta_2**(step+1))
                    self.params[k] -= (self.learning_rate/torch.sqrt(v_hat + self.epsilon))*m_hat

                if step == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = 0.999*smooth_loss + 0.001*loss
                losses.append(smooth_loss)

                step += 1
                hprev = hts

            print(f"\t * Smooth loss: {smooth_loss:.4f}")
            if (epoch + 1) % synth_interval == 0:
                _, s_syn = self.synthetize_seq(hprev[:, 0:1], X_train[:, 0, 0], 200, T)
                print("-" * 100)
                print(f"Synthesized sequence: \n{s_syn}")
                print("-" * 100)

        if save:
            with open(Path(f'./models/RNN/rnn_adam_{time.time()}.pickle'), 'wb') as handle:
                pickle.dump({
                    **self.params, 
                    'book_fname': self.book_fname, 
                    'hidden_size': self.hidden_size, 
                    'learning_rate': self.learning_rate, 
                    'epsilon': self.epsilon, 
                    'seq_length': self.seq_length
                    }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info("End of Adam training.")
        
        return losses
            
    def run(self, batch_size, n_epochs, synth_interval = 5, T = 1, optimizer = "adagrad", save = False, figure_filename = None):
        start_time = time.time()
        if optimizer == "adagrad":
            losses = self.train_adagrad(batch_size, n_epochs, synth_interval = synth_interval, T = T, save = save)
        elif optimizer == "adam":
            losses = self.train_adam(batch_size, n_epochs, synth_interval = synth_interval, T = T, save = save)
        else:
            logger.error("Unknown optimizer passed.")
            raise ValueError("Unknown optimizer")
        end_time = time.time()
        print(f'Time elapsed ({batch_size} samples per batch, {n_epochs} epochs): {(end_time - start_time):.1f} seconds.')
        
        plt.plot(losses)
        plt.xlabel('Steps')
        plt.ylabel('Smooth loss')
        plt.title(f'eta: {self.learning_rate} - seq_len: {self.seq_length} - m: {self.hidden_size} - epochs: {n_epochs} - batch_size: {batch_size}')
        plt.grid(True)
        if figure_filename:
            plt.savefig(Path(f"./reports/figures/{figure_filename}"))
        else:
            plt.show()
        
### TRAINING ###
if __name__ == '__main__':
    params = {
        'data_path': './data/shakespeare.txt',
        'hidden_size': 256,
        'seq_length': 100,
        'learning_rate': 0.001,
        }
    rnn = RNN(**params)
    rnn.run(64, 50, 5, optimizer="adam", save = True, figure_filename=f"run_{time.time()}.png")
###############