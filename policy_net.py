import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim1=64, hidden_dim2=32, seq_len=60):
        super(PolicyNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = seq_len
        # Define the LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim1,
            batch_first=True
            )
        self.dropout = nn.Dropout()
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim1,
            hidden_size=self.hidden_dim2,
            batch_first=True
            )
        self.linear = nn.Linear(self.hidden_dim2, 3)

    # assumed input size: batch_size * seq_len * input_dim
    def forward(self, input):
        output, _ = self.lstm1(input)
        output = self.dropout(output)
        output, (h, c) = self.lstm2(output)
        qvals = self.linear(h).squeeze(0)
        return qvals

    def get_actions(self, states):
        with torch.no_grad():
            action_values = self(states) # calls forward implicity
            actions = action_values.argmax(dim=1).type(torch.float32) - 1 # action in {-1, 0, 1}
        return actions
