import torch
import torch.nn as nn

class ActorNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=52, seq_len=60, drop_rate=0.5):
        super(ActorNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.drop_rate = drop_rate
        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=self.drop_rate,
            batch_first=True
            )
        self.relu = nn.ReLU()
        # policy function
        self.linear = nn.Linear(self.hidden_dim, 3)
        self.softmax = nn.Softmax(dim=1)
    # assumed input size: batch_size * seq_len * input_dim
    def forward(self, input):
        output, h = self.lstm(input)
        ff_input = h[0][1] # get second layer hidden output
        output = self.relu(ff_input)
        policy_dist = self.softmax(self.linear(output))
        return policy_dist
    def get_actions(self, states):
        with torch.no_grad():
            action_values = self(states) # calls forward implicity
            actions = action_values.argmax(dim=1).type(torch.float32) - 1 # action in {-1, 0, 1}
        return actions

class CriticNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=52, seq_len=60, drop_rate=0.5):
        super(CriticNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.drop_rate = drop_rate
        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=self.drop_rate,
            batch_first=True
            )
        self.relu = nn.ReLU()
        # value function
        self.linear = nn.Linear(self.hidden_dim, 1)

    # assumed input size: batch_size * seq_len * input_dim
    def forward(self, input):
        output, h = self.lstm(input)
        ff_input = h[0][1] # get second layer hidden output
        output = self.relu(ff_input)
        value = self.linear(output)
        return value
