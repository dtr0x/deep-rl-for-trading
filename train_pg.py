import random, torch
import numpy as np
from save_states_all import load_states
from policy_net import PolicyNet
from reinforcement import reward
import time

# set device for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
discount_factor = 0.3
learning_rate = 1e-4
num_epochs = 20
# for computing rewards (on-device)
bp = torch.tensor(2e-3, device=device)
tgt_vol = torch.tensor(0.1, device=device)

policy_net = PolicyNet().to(device)

optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

if __name__ == '__main__':
    # duration to train for, approximately 5 years 2010-2014
    train_idx = range(1000,2500)
    # load commodoties data
    S, P, sigall = [x.to(device) for x in load_states('commodity')]
    nidx = [7,16,20,25,27,32,35]
    idx = [i for i in np.arange(len(S)) if i not in nidx]
    S = S[idx]
    P = P[idx]
    sigall = sigall[idx]

    n_assets = len(S)
    n_steps = len(train_idx)

    actions_prev = torch.zeros(n_assets, device=device)

    for i_epoch in range(num_epochs):
        t1 = time.time()
        # Store rewards and actions for this trajectory (epoch)
        rewards_all = torch.zeros((n_steps, n_assets), dtype=torch.float32, device=device)
        actions_all = torch.zeros((n_steps, n_assets), dtype=torch.long, device=device)


        step = 0 # for indexing rewards/actions
        # loop to store trajectories
        for t in train_idx:
            # get states/prices/sigma values for each asset
            states = S[:, t]
            next_states = S[:, t+1]
            prices = P[:, t]
            prices_next = P[:, t+1]
            sigmas = sigall[:, t]


            # get actions for each asset
            actions = policy_net.get_actions(states)

            # compute rewards for each asset
            rewards = reward(prices, prices_next, sigmas, actions, actions_prev, tgt_vol, bp)

            actions_all[step] = actions
            rewards_all[step] = rewards

            actions_prev = actions
            step += 1

        step = 0 # for indexing rewards/actions
        # loop to optimize parameters
        for t in train_idx:
            # get states/prices/sigma values for each asset
            states = S[:, t]
            next_states = S[:, t+1]
            rewards = rewards_all[step:]
            actions = actions_all[step]

            gamma_exp = torch.tensor([discount_factor**i for i in range(len(rewards))], device=device)

            g = (gamma_exp.unsqueeze(1) * rewards).sum(axis=0)

            # get values of currrent actions from policy net
            action_values = policy_net(states)
            action_idx = actions + 1 # action indexes
            row_idx = torch.arange(action_values.size(0))
            action_values = action_values[row_idx, action_idx]
            log_action_values = action_values.log()

            policy_grad_op = discount_factor**step * g * log_action_values

            optimizer.zero_grad() # set gradients to 0 to avoid accumulation

            policy_grad_op.backward(torch.ones_like(policy_grad_op)) # backpropograte to compute gradient wrt parameters

            optimizer.step() # perform gradient descent step

            actions_prev = actions
            step += 1
        t2 = time.time()
        t = (t2-t1)/60
        print("Finished epoch {} in {:.2f} minutes".format(i_epoch+1, t))
        # save the model every epoch
        torch.save(policy_net.state_dict(), "models/pg_epoch_{}.pt".format(i_epoch+1))
