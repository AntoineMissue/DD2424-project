import torch
import matplotlib.pyplot as plt
import time
import tqdm

from nlpProject.one_layer_lstm import LSTM1
from nlpProject.make_data import DataMaker
from nlpProject.utils import compute_loss, make_dataloader
from nlpProject.inference import synthesize_seq_lstm1

def train_lstm1(data_path, n_epochs, hidden_size, seq_length, batch_size, learning_rate, synth_interval = 5, fig_savepath = None):
    losses = []
    step = 0
    smooth_loss = 0
    data_maker = DataMaker(data_path)
    _, input_size, _, _ = data_maker.make_charmap()
    model = LSTM1(input_size, hidden_size, input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = make_dataloader(data_maker, seq_length, batch_size)

    model.train()
    start_time = time.time()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs} - Step {step + 1}")
        h_prev, c_prev = (torch.zeros(model.hidden_size, batch_size, dtype=torch.double),
                          torch.zeros(model.hidden_size, batch_size, dtype=torch.double))
        
        for X_train, Y_train in tqdm.tqdm(dataloader, desc="Processing batches"):
            X_train = X_train.permute(2, 1, 0)
            Y_train = Y_train.permute(2, 1, 0)

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

            h_prev.detach_()
            c_prev.detach_()

            step += 1

        print(f"\t * Smooth loss: {smooth_loss:.4f}")
        if (epoch + 1) % synth_interval == 0:
            _, s_syn = synthesize_seq_lstm1(model, data_path, h_prev[:, 0:1], c_prev[:, 0:1], X_train[:, 0, 0], 200)
            print("-" * 100)
            print(f"Synthesized sequence: \n{s_syn}")
            print("-" * 100)

    end_time = time.time()
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
    train_lstm1(data_path='./data/shakespeare.txt', 
                n_epochs=50, 
                hidden_size=256, 
                seq_length=100, 
                batch_size=64, 
                learning_rate=0.001,
                synth_interval=5,
                fig_savepath= './reports/figures/lstm_1_layer_test')