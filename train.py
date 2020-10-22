from util import *
from state_space import state_space
from policy_net import PolicyNet
from replay_memory import ReplayMemory
from reinforcement import *
import random

# Parameters
discount_factor = 0.3
bp = 2e-3
replay_capacity = 5000
target_update = 1000
learning_rate = 1e-4
num_epochs = 20
batch_size = 64

# the minimum day index we can use to compute a state
min_day_idx = 312

# epsilon greedy policy
eps_start = 0.95
eps_end = 0.1
eps_len = int(num_epochs/2) # number of epochs to decay epsilon

# linear annealing of epsilon
eps_sched = np.linspace(eps_start, eps_end, eps_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = PolicyNet().to(device)
target_net = PolicyNet().to(device)
target_net.eval()  # no optimization is performed directly on target network

def select_action(states, eps):
    # TODO: modify this function. The epsilon-greedy policy should be applied asset-wise
    sample = random.random()
    if sample > eps:
        # select best action from model with probability 1-epsilon
        with torch.no_grad():
            actions = policy_net.forward(states)
    else:
        # return random actions
        actions = np.random.choice([-1, 0, 1], len(states))
        actions = torch.tensor(actions, device=device)
    return actions

# optimizer / memory
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(replay_capacity)

if __name__ == '__main__':
    # test commodoties data
    # data = pd.read_csv('cleaned_data.csv').drop(columns='date')
    # cols = [c for c in data.columns if 'Comdty' in c]
    # data = data[cols]
    # states = state_space(data)

    # RANDOM DATA PLACEHOLDER
    n_assets = 10
    n_days = 1000
    seq_length = 60
    features = 7
    states = torch.rand(n_assets, n_days, seq_length, features)

    price_data = df2tensor(pd.load('cleaned_data.csv')).drop(columns='date')
    price_data = price_data[min_day_idx:] # get price days to correspond with states
    T = len(price_data) - 1

    # store previous actions for each asset, initially 0
    actions_prev = torch.zeros(len(states))

    for i_epoch in range(num_epochs):
        # epsilon greedy action selection
        if i_epoch < eps_len:
            eps = eps_sched[i_epoch]
        else:
            eps = eps_end

        for t in range(T):
            ast_state = states[:, t, :, :] # .squeeze() ? get states for all assets at time t
            ast_price = price_data[t, :] # get corresponding prices for all assets at time t
            actions = select_action(states, eps) # get actions for each asset

            # TODO: implement get_reward function
            next_prices = price_data[t+1, :]
            rewards = get_reward(next_prices, actions, actions_prev)
            rewards = torch.tensor([rewards], device=device)

            next_states = states[:, t+1, :, :]
            # TODO: push each transition to replay memory. Verify dimensions
            memory.push(state, action, next_state, reward)

            # TODO: optimize after each action
            optimize_model(optimizer, memory, policy_net, target_net, batch_size, discount_factor)

            # update target network after target_update steps
            if t % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

    # TODO: save model, track loss function and implement early stopping.
