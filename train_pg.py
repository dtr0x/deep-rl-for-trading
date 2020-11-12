import random, torch
import numpy as np
from save_states_all import load_states
from policy_net import PolicyNet
from reinforcement import get_reward
from torch.nn.functional import softmax
import time, argparse

# set device for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_pg(policy_net, optimizer, asset_type, min_year=2006, max_year=2010,
    val_frac=0.1, discount_factor=0.3, n_epochs=20, bp=0.002, tgt_vol=0.1, num_epochs=20):
    # durations to train/validate for
    min_year_idx = min_year % 2006 + 1
    max_year_idx = max_year % 2006 + 1
    min_idx = 252*(min_year_idx-1)
    max_idx = 252*max_year_idx - 1
    val_idx_min = int((1-val_frac) * (max_idx-min_idx)) + min_idx
    train_idx = range(min_idx, val_idx_min)
    val_idx = range(val_idx_min, max_idx)

    n_steps = len(train_idx)

    # load data
    S, P, sigall = [x.to(device) for x in load_states(asset_type)]
    n_assets = len(S)

    actions_prev = torch.zeros(n_assets, device=device)

    for i_epoch in range(num_epochs):
        policy_net.train()
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
            sigmas = sigall[:, t+1]
            sigmas_prev = sigall[:, t]

            # get actions for each asset
            actions = policy_net.get_actions(states)

            # compute rewards for each asset
            rewards = get_reward(prices, prices_next, sigmas, sigmas_prev, actions, actions_prev, tgt_vol, bp)

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
            probs = softmax(action_values, dim=1)
            action_idx = actions.long() + 1 # action indexes
            row_idx = torch.arange(probs.size(0))
            probs = probs[row_idx, action_idx]
            log_probs = probs.log()

            policy_grad_op = discount_factor**step * g * log_probs

            optimizer.zero_grad() # set gradients to 0 to avoid accumulation

            policy_grad_op.sum().backward() # backpropograte to compute gradient wrt parameters

            optimizer.step() # perform gradient descent step

            actions_prev = actions
            step += 1

        # validation loop
        val_returns = []
        actions_prev = torch.zeros(len(S), device=device)
        policy_net.eval() # eval mode for validation
        for t in val_idx:
            # get states/prices/sigma values for each asset
            states = S[:, t]
            next_states = S[:, t+1]
            prices = P[:, t]
            prices_next = P[:, t+1]
            sigmas = sigall[:, t+1]
            sigmas_prev = sigall[:, t]
            actions = policy_net.get_actions(states) # get actions from DQN
            # compute rewards
            rewards = get_reward(prices, prices_next, sigmas, sigmas_prev,
            actions, actions_prev, tgt_vol, bp)
            val_returns.append(rewards.mean().item())
            # store actions for next time step to compute reward
            actions_prev = actions
        print("Last actions chosen on validation set:", actions)

        # compare the average reward received on val set for early stopping
        avg_val_return = np.mean(val_returns)
        print("Average return on validation set at epoch {}: {:.3f}".format(i_epoch, avg_val_return))
        if i_epoch == 0:
            best_return = avg_val_return
            torch.save(policy_net.state_dict(), "models/pg_{}_{}_{}.pt".format(asset_type, min_year, max_year))
        elif avg_val_return > best_return:
            best_return = avg_val_return
            torch.save(policy_net.state_dict(), "models/pg_{}_{}_{}.pt".format(asset_type, min_year, max_year))

        t2 = time.time()
        t = (t2-t1)/60
        av_ret_epoch = rewards_all.mean()
        print("Finished epoch {} in {:.2f} minutes. Average return: {:.3f}".format(i_epoch, t, av_ret_epoch))
        i_epoch += 1

if __name__ == '__main__':
    # get arguments from command line for asset type and year range
    parser = argparse.ArgumentParser(description='Select asset type and year range to train.')
    parser.add_argument('-asset_type', type=str, required=True)
    parser.add_argument('-min_year', type=int, required=True)
    parser.add_argument('-max_year', type=int, required=True)
    args = parser.parse_args()
    asset_type = args.asset_type
    min_year = args.min_year
    max_year = args.max_year

    # set up networks, memory, optimizer
    policy_net = PolicyNet().to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    print("Training Policy Gradient on {} assets for years {} to {}...".format(asset_type, min_year, max_year))

    # train and save best params
    train_pg(policy_net, optimizer, asset_type, min_year, max_year)
