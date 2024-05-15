import torch
import matplotlib.pyplot as plt
import time

from nlpProject.one_layer_lstm import LSTM1
from nlpProject.make_data import DataMaker
from nlpProject.utils import compute_loss
from nlpProject.inference import synthesize_seq_lstm1

def train_lstm1(data_path, n_epochs, hidden_size, seq_length, batch_size, learning_rate, fig_savepath = None):
    losses = []
    e, step, epoch = 0, 0, 0
    smooth_loss = 0
    data_maker = DataMaker(data_path)
    data, input_size, _, _ = data_maker.make_charmap()
    model = LSTM1(input_size, hidden_size, input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    h_prev, c_prev = (torch.zeros(model.hidden_size, batch_size, dtype=torch.double),
                      torch.zeros(model.hidden_size, batch_size, dtype=torch.double))

    model.train()
    start_time = time.time()
    while epoch < n_epochs:
        X_batch = []
        Y_batch = []
        for b in range(batch_size):
            start_index = e + b * seq_length
            X_chars = data[start_index:(start_index + seq_length)]
            Y_chars = data[(start_index + 1):(start_index + seq_length + 1)]
            X_batch.append(data_maker.encode_string(X_chars))
            Y_batch.append(data_maker.encode_string(Y_chars))

        X_train = torch.stack(X_batch, dim=2)
        Y_train = torch.stack(Y_batch, dim=2)

        P_train, (h_prev, c_prev) = model(X_train, init_states = (h_prev, c_prev))
        loss = compute_loss(Y_train, P_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            smooth_loss = loss.item()
        else:
            smooth_loss = 0.999*smooth_loss + 0.001*loss.item()
        losses.append(smooth_loss)

        if step % 1000 == 0:
            print(f"Step: {step}")
            print(f"\t * Smooth loss: {smooth_loss:.4f}")
        if step % 5000 == 0:
            _, s_syn = synthesize_seq_lstm1(model, data_path, h_prev[:, 0:1], c_prev[:, 0:1], X_train[:, 0, 0], 200)
            print("-" * 100)
            print(f"Synthesized sequence: \n{s_syn}")
            print("-" * 100)
        if step % 100000 == 0 and step > 0:
            _, s_lsyn = synthesize_seq_lstm1(model, data_path, h_prev[:, 0:1], c_prev[:, 0:1], X_train[:, 0, 0], 1000)
            print("-" * 100)
            print(f"Long synthesized sequence: \n{s_lsyn}")
            print("-" * 100)

        step += 1
        e += batch_size * seq_length
        if e > len(data) - batch_size * seq_length:
            e = 0
            epoch += 1
            h_prev, c_prev = (torch.zeros(model.hidden_size, batch_size, dtype=torch.double),
                              torch.zeros(model.hidden_size, batch_size, dtype=torch.double))
        else:
            h_prev.detach_()
            c_prev.detach_()
    end_time = time.time()
    duration = end_time - start_time
    _, s_lsyn = synthesize_seq_lstm1(model, data_path, h_prev[:, 0:1], c_prev[:, 0:1], X_train[:, 0, 0], 5000)
    print("-" * 100)
    print(f"Benchmark synthesized sequence: \n{s_lsyn}")
    print("-" * 100)
    print(f'No. of steps: {step + 1}.')
    print(f'Time elapsed ({batch_size} samples per batch, {n_epochs} epochs): {(end_time - start_time):.1f} seconds.')

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Smooth loss')
    plt.title(f'eta: {learning_rate} - seq_len: {seq_length} - m: {hidden_size} - epochs: {n_epochs} - batch_size: {batch_size}')
    plt.grid(True)
    if fig_savepath:
        plt.savefig(fig_savepath)
    else:
        plt.show()

if __name__ == '__main__':
    train_lstm1('./data/shakespeare.txt', 50, 256, 100, 64, 0.001, './reports/figures/lstm_1_layer_test')