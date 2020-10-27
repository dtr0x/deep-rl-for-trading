import torch
import torch.nn as nn

# Parameters
input_dim = 7
hidden_dim1 = 64
hidden_dim2 = 32
drop_rate = 0.2
seq_len = 60

class PolicyNet(nn.Module):
    def __init__(self, is_dqn=True):
        super(PolicyNet, self).__init__()
        self.is_dqn = is_dqn

        # Define the first LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1)

        # dropout some outputs to avoid overfitting
        self.dropout = nn.Dropout(drop_rate)

        # apply leaky ReLU activation to outputs of dropout
        self.leaky_relu = nn.LeakyReLU()

        # Define the first LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2)

        if self.is_dqn:
            # discrete action space
            self.linear = nn.Linear(hidden_dim2, 3)
            self.softmax = nn.Softmax(dim=1)
        else:
            # continuous action space
            self.linear = nn.Linear(hidden_dim2, 1)
            self.tanh = nn.Tanh()

    # define forward pass through LSTM with single output in (-1, 1)
    # assumed input size: batch_size * seq_len * input_dim
    def forward(self, input):
        # reshape input for LSTM input dimensions
        i = input.view(seq_len, -1, input_dim)

        output, _ = self.lstm1(i)
        # print("lstm layer 1 output size: ", output.size())

        output = self.dropout(output)
        # print("dropout layer output size: ", output.size())

        output = self.leaky_relu(output)
        # print("relu layer output size: ", output.size())

        # the output of the last LSTM in the chain (h) is passed to linear
        output, h = self.lstm2(output)
        # print("lstm layer 2 output size: ", output.size())

        ff_input = h[0].squeeze()
        # print("ff input size: ", ff_input.size())

        output = self.linear(ff_input)
        # ("linear layer output size: ", output.size())

        if self.is_dqn:
            output = self.softmax(output)
        else:
            # scale output to (-1, 1)
            output = self.tanh(output)

        return output

if __name__ == '__main__':
    # test input/output dims
    batch_size = 64
    t = torch.randn(batch_size, seq_len, input_dim)
    dqn = PolicyNet()
    out = dqn(t)

    print(out)
