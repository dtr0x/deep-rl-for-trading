import torch
import torch.nn as nn

# Parameters
input_dim = 7
hidden_dim = 52
drop_rate = 0.5
seq_len = 60

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        # Define the first LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=drop_rate,
            batch_first=True
            )

        # discrete action space
        self.linear = nn.Linear(hidden_dim, 3)
        self.softmax = nn.Softmax(dim=1)

    # assumed input size: batch_size * seq_len * input_dim
    def forward(self, input):
        output, h = self.lstm(input)

        ff_input = h[0][1] # get second layer hidden output

        output = self.linear(ff_input)

        output = self.softmax(output)

        return output

    def get_actions(self, states):
        with torch.no_grad():
            action_values = self(states) # calls forward implicity
            actions = action_values.argmax(dim=1).type(torch.long) - 1 # action in {-1, 0, 1}
        return actions
