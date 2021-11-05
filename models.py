# This file defines the model used in the task

from torch import nn
from config import BATCH_FIRST


class AntiSpoofingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, LSTM_num_layers, linear_size, output_size):
        super(AntiSpoofingRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, LSTM_num_layers, batch_first=BATCH_FIRST)
        self.l1 = nn.Linear(hidden_size, linear_size)
        self.l2 = nn.Linear(linear_size, output_size)

    def forward(self, input, h0, c0):
        output, (_, _) = self.rnn(input, (h0, c0))
        x = self.l1(output[:, -1, :])
        x = self.l2(x)
        x = nn.Sigmoid()(x)
        return x
