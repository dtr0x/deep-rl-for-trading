import random, torch
import numpy as np
from save_states_all import load_states
from policy_net import PolicyNet
from replay_memory import ReplayMemory
from reinforcement import reward
from optimization import optimize_model

# set device for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
discount_factor = 0.3
replay_capacity = 5000
target_update = 1000
learning_rate = 1e-4
num_epochs = 20
batch_size = 64
# for computing rewards (on-device)
bp = torch.tensor(2e-3, device=device)
tgt_vol = torch.tensor(0.1, device=device)

# epsilon greedy policy
eps_start = 0.9
eps_end = 0.1
eps_len = int(num_epochs/2) # number of epochs to decay epsilon

# linear annealing of epsilon
eps_sched = np.linspace(eps_start, eps_end, eps_len)

policy_net = PolicyNet().to(device)
target_net = PolicyNet().to(device)
target_net.eval()  # no optimization is performed directly on target network

# The epsilon-greedy policy is applied applied asset-wise
def select_action(states, eps):
    sample = random.random()
    if sample > eps:
        # select best action from model with probability 1-epsilon
        actions = policy_net.get_actions(states)
    else:
        # return random actions
        actions = np.random.choice([-1, 0, 1], len(states))
        actions = torch.tensor(actions, dtype=torch.long, device=device)
    return actions

# optimizer / memory
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(replay_capacity)

if __name__ == '__main__':
    # duration to train for, approximately 5 years 2010-2014
    train_idx = range(1000,2000)
    # load commodoties data
    S, P, sigall = [x.to(device) for x in load_states('commodity')]

    # store previous actions for each asset, initially 0
    actions_prev = torch.zeros(len(S), device=device)

    step = 1 # keep track of all iterations to update target network when step % target_update == 0
    losses = [] # keep track of losses, report average loss every 100 steps

    for i_epoch in range(num_epochs):
        # epsilon greedy action selection
        if i_epoch < eps_len:
            eps = eps_sched[i_epoch]
        else:
            eps = eps_end

        for t in train_idx:
            # get states/prices/sigma values for each asset
            states = S[:, t].contiguous() # contiguous in memory to call .view() function
            next_states = S[:, t+1].contiguous()
            prices = P[:, t]
            prices_next = P[:, t+1]
            sigmas = sigall[:, t]
            actions = select_action(states, eps) # get actions for each asset

            # compute rewards
            rewards = reward(prices, prices_next, sigmas, actions, actions_prev, tgt_vol, bp)

            # push individual transitions to replay memory
            for s, a, ns, r in zip(states, actions, next_states, rewards):
                memory.push(s, a, ns, r)

            # Perform gradient descent at each time step
            loss = optimize_model(optimizer, memory, policy_net, target_net, batch_size, discount_factor)
            if loss:
                losses.append(loss.item())

            # print average loss every 100 steps
            if step % 100 == 0:
                avg_loss = np.mean(losses)
                print("Average loss after step {}: {:.3f}".format(step, avg_loss))
                losses = []

            # update target network after target_update steps
            if step % target_update == 0:
                 target_net.load_state_dict(policy_net.state_dict())

            # store actions for next time step to compute reward
            actions_prev = actions

            step += 1

        # save the model every epoch
        torch.save(target_net.state_dict(), "models/dqn_epoch_{}.pt".format(i_epoch+1))
