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

# if __name__ == '__main__':
#     import torch
#     from torch import nn
#     from torch.utils.data import DataLoader
#     from datasets import ReplySpoofDataset, collate_fn_pad
#     from config import INPUT_SIZE, HIDDEN_SIZE, LINEAR_SIZE, LSTM_NUM_LAYERS, BATCH_SIZE, OUTPUT_SIZE, VAL_INDEX
#
#     net = AntiSpoofingRNN(INPUT_SIZE, HIDDEN_SIZE, LSTM_NUM_LAYERS, LINEAR_SIZE, OUTPUT_SIZE)
#     dataset = ReplySpoofDataset(VAL_INDEX)
#     train_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn_pad)
#     for x, _ in train_loader:
#         input = x
#         break
#     h0 = torch.randn(LSTM_NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)  # D * num_layers, N, H_out
#     c0 = torch.randn(LSTM_NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
#     out = net(input, h0, c0)
#     print(out[0])
