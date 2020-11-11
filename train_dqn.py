import torch
import numpy as np
from save_states_all import load_states
from policy_net import PolicyNet
from replay_memory import ReplayMemory
from reinforcement import get_reward
from optimization import optimize_model
import time, argparse

# set device for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select action using epsilon-greedy policy
def select_action(policy_net, state, eps):
    sample = np.random.rand()
    if sample > eps:
        # select best action from model with probability 1-epsilon
        action = policy_net.get_actions(state.unsqueeze(0))
    else:
        # return random action
        action = np.random.choice([-1, 0, 1])
        action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    return action

# main training function
# Input:
#   policy_net: network to optimize during training
#   target_net: network to update after target_update iterations
#   memory: replay memory
#   optimizer: optimization scheme (gradient descent)
#   asset type: 'all', 'commodity', 'currency', or 'index'
#   min_year: initial year to start training (2006-2019)
#   max_year: year to end training (> min_year, 2007-2020)
#   val_frac: the fraction of training data to use for validation
#   discount_factor: gamma for Q-learning
#   target_update: number of iterations until target network is updated
#   batch_size: number of transitions to pass to optimizer at each iteration
#   bp: penalty for trade commission (see reward function)
#   tgt_vol: target volatility (see reward function)
#   n_epochs:
def train_dqn(policy_net, target_net, memory, optimizer, asset_type, min_year=2006,
    max_year=2010, val_frac=0.1, discount_factor=0.3, target_update=1000, n_epochs=20,
    batch_size=64, bp=0.002, tgt_vol=0.1):
    # durations to train/validate for
    min_year_idx = min_year % 2006 + 1
    max_year_idx = max_year % 2006 + 1
    min_idx = 252*(min_year_idx-1)
    max_idx = 252*max_year_idx - 1
    val_idx_min = int((1-val_frac) * (max_idx-min_idx)) + min_idx
    train_idx = range(min_idx, val_idx_min)
    val_idx = range(val_idx_min, max_idx)

    # set target to evaluation mode (no dropout etc)
    target_net.eval()

    # for epsilon greedy action selection
    eps = 0.1

    # load data
    S, P, sigall = [x.to(device) for x in load_states(asset_type)]
    n_assets = len(S)

    step = 0 # keep track of all iterations to update target network when step % target_update == 0
    losses = []
    for i_epoch in range(n_epochs):
        policy_net.train() # call eval before validating, reset to train mode at every epoch
        epoch_rewards = []
        t1 = time.time()
        # store previous action for asset, initially 0
        actions_prev = torch.zeros(n_assets, device=device)
        # training loop
        for t in train_idx:
            for i in range(n_assets):
                state = S[i, t]
                next_state = S[i, t+1]
                price = P[i, t]
                next_price = P[i, t+1]
                sigma = sigall[i, t+1]
                sigma_prev = sigall[i, t] # sigma at PREVIOUS time step
                action = select_action(policy_net, state, eps) # get actions from policy
                # compute rewards
                reward = get_reward(price, next_price, sigma, sigma_prev,
                    action, actions_prev[i], tgt_vol, bp)
                epoch_rewards.append(reward.item())
                # store transition for each asset and optimize
                memory.push(state, action, next_state, reward)
                loss = optimize_model(optimizer, memory, policy_net, target_net,
                    batch_size, discount_factor)
                if loss:
                    losses.append(loss.item())
                # update target network after target_update steps
                if (step + 1) % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    avg_loss = np.mean(losses)
                    print("Average loss over last 100 steps at step {}: {:.3f}".format(step+1, avg_loss))
                    losses = []
                step += 1
                # set previous actions to current actions
                actions_prev[i] = action

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
            torch.save(policy_net.state_dict(), "models/dqn_{}_{}_{}.pt".format(asset_type, min_year, max_year))
        elif avg_val_return > best_return:
            best_return = avg_val_return
            torch.save(policy_net.state_dict(), "models/dqn_{}_{}_{}.pt".format(asset_type, min_year, max_year))

        t2 = time.time()
        t = (t2-t1)/60
        av_ret_epoch = np.mean(epoch_rewards)
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
    target_net = PolicyNet().to(device)
    memory = ReplayMemory(5000)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    print("Training DQN on {} assets for years {} to {}...".format(asset_type, min_year, max_year))

    # train and save best params
    train_dqn(policy_net, target_net, memory, optimizer, asset_type, min_year, max_year)
