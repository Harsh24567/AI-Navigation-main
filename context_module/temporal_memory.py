import torch
import torch.nn as nn

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=16, output_dim=7, num_layers=1):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.linear(last_out)
        return self.softmax(logits)
