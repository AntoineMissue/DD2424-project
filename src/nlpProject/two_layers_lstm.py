import torch
import torch.nn as nn

class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True).double()
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True).double()
        self.fc = nn.Linear(hidden_size, output_size).double()

    def forward(self, x, init_states=None, T=1.0):
        if init_states is None:
            h0_1 = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float64).to(x.device)
            c0_1 = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float64).to(x.device)
            h0_2 = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float64).to(x.device)
            c0_2 = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float64).to(x.device)
        else:
            (h0_1, c0_1), (h0_2, c0_2) = init_states

        out, (h1, c1) = self.lstm1(x, (h0_1, c0_1))
        out, (h2, c2) = self.lstm2(out, (h0_2, c0_2))
        out = self.fc(out)
        P = torch.softmax(out/T, dim = 2)
        return P, ((h1, c1), (h2, c2))
