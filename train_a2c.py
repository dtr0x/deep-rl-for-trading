import random, torch
import numpy as np
from save_states_all import load_states
from actor_critic import ActorNet, CriticNet
from torch.nn.functional import mse_loss
from reinforcement import get_reward
import time, argparse

# set device for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select action by sampling from policy
def sample_action(actor_net, state):
    probs = actor_net(state.unsqueeze(0))
    u = torch.rand(1, device=device) # sample uniform value
    cdf = probs.cumsum(dim=1)
    if u <= cdf[0,0]:
        action_idx = 0
    elif u <= cdf[0,1]:
        action_idx = 1
    else:
        action_idx = 2
    return action_idx-1, probs[0, action_idx]

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
def train_a2c(actor_net, critic_net, actor_optimizer, critic_optimizer, asset_type, min_year=2006,
    max_year=2010, val_frac=0.1, discount_factor=0.3, batch_size=128, bp=0.002, tgt_vol=0.1, n_epochs=10):
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

    action_batch = torch.zeros((batch_size, n_assets), dtype=torch.float32, device=device)
    reward_batch = torch.zeros((batch_size, n_assets), dtype=torch.float32, device=device)
    value_batch = torch.zeros((batch_size, n_assets), dtype=torch.float32, device=device)
    advantage_batch = torch.zeros((batch_size, n_assets), dtype=torch.float32, device=device)
    logprob_batch = torch.zeros((batch_size, n_assets), dtype=torch.float32, device=device)

    for i_epoch in range(n_epochs):
        t1 = time.time()
        actions_prev = torch.zeros(n_assets, dtype=torch.float32, device=device)
        for t in train_idx:
            batch_idx = (t - min_idx) % batch_size
            for i in range(n_assets):
                state = S[i, t]
                next_state = S[i, t+1]
                price = P[i, t]
                next_price = P[i, t+1]
                sigma = sigall[i, t].unsqueeze(0)
                a, prob = sample_action(actor_net, state)
                a_prev = actions_prev[i] # previous action
                reward = get_reward(price, next_price, sigma, a, a_prev, tgt_vol, bp)
                actions_prev[i] = a

                value = critic_net(state.unsqueeze(0))
                advantage = reward + discount_factor * critic_net(next_state.unsqueeze(0)) - value

                action_batch[batch_idx, i] = a
                reward_batch[batch_idx, i] = reward
                value_batch[batch_idx, i] = value
                advantage_batch[batch_idx, i] = advantage
                logprob_batch[batch_idx, i] = prob.log()

            # batch full, do optimization
            if batch_idx == batch_size - 1:
                # optimize actor network
                gamma = torch.tensor([discount_factor**i for i in range(batch_size)], device=device)
                actor_loss = (gamma.unsqueeze(0).transpose(0,1) * (logprob_batch * advantage_batch.detach())).sum(axis=0).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # optimize critic network
                target = advantage_batch + value_batch
                critic_loss = mse_loss(target, value_batch)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                logprob_batch.detach_()
                advantage_batch.detach_()
                value_batch.detach_()

                print("Actor loss: {:.3f}".format(actor_loss.item()))
                print("Critic loss: {:.3f}".format(critic_loss.item()))

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
            actions = actor_net.get_actions(states) # get actions from DQN
            # compute rewards
            rewards = get_reward(prices, prices_next, sigmas, actions, actions_prev, tgt_vol, bp)
            val_returns.append(rewards.mean().item())
            # store actions for next time step to compute reward
            actions_prev = actions
        print("Last actions chosen on validation set:", actions)
        # compare the average reward received on val set for early stopping
        avg_val_return = np.mean(val_returns)
        print("Average return on validation set at epoch {}: {:.3f}".format(i_epoch, avg_val_return))

        t2 = time.time()
        t = (t2-t1)/60
        print("Finished epoch {} in {:.2f} minutes".format(i_epoch, t))

    return actor_net.state_dict()

if __name__ == '__main__':
    get arguments from command line for asset type and year range
    parser = argparse.ArgumentParser(description='Select asset type and year range to train.')
    parser.add_argument('-asset_type', type=str, required=True)
    parser.add_argument('-min_year', type=int, required=True)
    parser.add_argument('-max_year', type=int, required=True)
    args = parser.parse_args()
    asset_type = args.asset_type
    min_year = args.min_year
    max_year = args.max_year

    # set up networks, memory, optimizer
    actor_net = ActorNet().to(device)
    critic_net = CriticNet().to(device)
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=1e-3)

    print("Training A2C on {} assets for years {} to {}...".format(asset_type, min_year, max_year))

    # train and save best params
    actor_params = train_a2c(actor_net, critic_net, actor_optimizer, critic_optimizer, asset_type, min_year, max_year)
    torch.save(actor_params, "models/a2c_{}_{}_{}.pt".format(asset_type, min_year, max_year))
