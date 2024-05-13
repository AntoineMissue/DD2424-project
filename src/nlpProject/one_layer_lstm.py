import torch
import torch.nn as nn

class LSTM1(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_size = input_size

        self.Wf = nn.Parameter(torch.randn(hidden_size, input_size, dtype=torch.double))
        self.Wi = nn.Parameter(torch.randn(hidden_size, input_size, dtype=torch.double))
        self.Wo = nn.Parameter(torch.randn(hidden_size, input_size, dtype=torch.double))
        self.Wc = nn.Parameter(torch.randn(hidden_size, input_size, dtype=torch.double))
        self.Uf = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.double))
        self.Ui = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.double))
        self.Uo = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.double))
        self.Uc = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.double))
        self.bf = nn.Parameter(torch.zeros(hidden_size, dtype=torch.double))
        self.bi = nn.Parameter(torch.zeros(hidden_size, dtype=torch.double))
        self.bo = nn.Parameter(torch.zeros(hidden_size, dtype=torch.double))
        self.bc = nn.Parameter(torch.zeros(hidden_size, dtype=torch.double))

        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.double)

    def forward(self, X, init_states=None):
        _, seq_len, batch_size = X.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size, batch_size, dtype=torch.double),
                        torch.zeros(self.hidden_size, batch_size, dtype=torch.double))
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = X[:, t, :]

            f_t = torch.sigmoid(self.Wf @ x_t + self.Uf @ h_t + self.bf[:, None])
            i_t = torch.sigmoid(self.Wi @ x_t + self.Ui @ h_t + self.bi[:, None])
            o_t = torch.sigmoid(self.Wo @ x_t + self.Uo @ h_t + self.bo[:, None])
            c_hat_t = torch.tanh(self.Wc @ x_t + self.Uc @ h_t + self.bc[:, None])

            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=1)
        hidden_seq = hidden_seq.reshape(self.hidden_size, seq_len, batch_size)
        output = self.fc(hidden_seq.permute(2, 1, 0)).permute(2, 1, 0)
        P = torch.softmax(output, dim = 0)
        return P, (h_t, c_t)