import random, torch
import numpy as np
from save_states_all import load_states
from policy_net import PolicyNet
from replay_memory import ReplayMemory
from reinforcement import get_reward
from optimization import optimize_model
import time, argparse

# set device for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon greedy policy
eps_start = 0.9
eps_end = 0.1
eps_len = 10 # number of epochs to decay epsilon

# linear annealing of epsilon
eps_sched = np.linspace(eps_start, eps_end, eps_len)

# Select action using epsilon-greedy policy
def select_action(policy_net, states, eps):
    sample = random.random()
    if sample > eps:
        # select best action from model with probability 1-epsilon
        actions = policy_net.get_actions(states)
    else:
        # return random actions
        actions = np.random.choice([-1, 0, 1], len(states))
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
    return actions

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
    max_year=2010, val_frac=0.1, discount_factor=0.3, target_update=10000,
    batch_size=64, bp=0.002, tgt_vol=0.1):
    # durations to train/validate for
    min_year = min_year % 2006 + 1
    max_year = max_year % 2006 + 1
    min_idx = 252*(min_year-1)
    max_idx = 252*max_year - 1
    val_idx_min = int((1-val_frac) * (max_idx-min_idx)) + min_idx
    train_idx = range(min_idx, val_idx_min)
    val_idx = range(val_idx_min, max_idx)

    # load data
    S, P, sigall = [x.to(device) for x in load_states(asset_type)]
    n_assets = len(S)

    # flag for early stopping
    early_stop = False
    step = 1 # keep track of all iterations to update target network when step % target_update == 0
    i_epoch = 1
    losses = []
    while not early_stop:
        t1 = time.time()
        # store optimal parameters for early stopping
        best_params = policy_net.state_dict()
        # epsilon greedy action selection
        if i_epoch <= eps_len:
            eps = eps_sched[i_epoch-1]
        else:
            eps = eps_end
        # training loop
        for i in range(n_assets):
            states = S[i]
            prices = P[i]
            sigmas = sigall[i]
            # store previous action for asset, initially 0
            action_prev = torch.zeros(1, device=device)
            for t in train_idx:
                action = select_action(policy_net, states[t].unsqueeze(0), eps) # get actions from DQN
                # compute rewards
                reward = get_reward(prices[t], prices[t+1], sigmas[t].unsqueeze(0),
                    action, action_prev, tgt_vol, bp)
                # store transition for each asset and optimize
                memory.push(states[t], action, states[t+1], reward)
                loss = optimize_model(optimizer, memory, policy_net, target_net,
                    batch_size, discount_factor)
                if loss:
                    losses.append(loss.item())
                    # update target network after target_update steps
                if step % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    avg_loss = np.mean(losses)
                    print("Average loss after step {}: {:.3f}".format(step, avg_loss))
                    losses = []
                step += 1
                # set previous actions to current actions
                action_prev = action

        # validation loop
        val_returns = []
        actions_prev = torch.zeros(len(S), device=device)
        for t in val_idx:
            # get states/prices/sigma values for each asset
            states = S[:, t]
            next_states = S[:, t+1]
            prices = P[:, t]
            prices_next = P[:, t+1]
            sigmas = sigall[:, t]
            actions = policy_net.get_actions(states) # get actions from DQN
            # compute rewards
            rewards = get_reward(prices, prices_next, sigmas, actions, actions_prev, tgt_vol, bp)
            val_returns.append(rewards.mean().item())
            # store actions for next time step to compute reward
            actions_prev = actions
        print("Last actions chosen on validation set:", actions)

        # compare the average reward received on val set for early stopping
        avg_val_return = np.mean(val_returns)
        print("Average return on validation set at epoch {}: {:.3f}".format(i_epoch, avg_val_return))
        if i_epoch == 15:
            best_val_return = avg_val_return
        elif i_epoch > 15:
            if avg_val_return > best_val_return:
                best_val_return = avg_val_return
            else:
                if best_val_return > 0:
                    early_stop = True

        t2 = time.time()
        t = (t2-t1)/60
        print("Finished epoch {} in {:.2f} minutes".format(i_epoch, t))
        i_epoch += 1

    print("Stopped training after epoch {}".format(i_epoch-1))
    # return best_params
    return best_params

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
    target_net.eval()  # no optimization is performed directly on target network
    memory = ReplayMemory(5000)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    print("Training DQN on {} assets for years {} to {}...".format(asset_type, min_year, max_year))

    # train and save best params
    params = train_dqn(policy_net, target_net, memory, optimizer, asset_type, min_year, max_year)
    torch.save(params, "models/dqn_{}_{}_{}.pt".format(asset_type, min_year, max_year))
