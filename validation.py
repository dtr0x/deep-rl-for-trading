import torch
from policy_net import PolicyNet
from save_states_all import load_states
from reinforcement import get_reward
import numpy as np
import matplotlib.pyplot as plt

def signR(prices, t):
    rets = (prices[:, t-251:t+1] - prices[:, t-252:t])/prices[:, t-252:t]
    return rets.mean(axis=1).sign()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyNet().to(device)

    asset_type = 'index'
    min_year = 2011
    max_year = 2015

    min_year = min_year % 2006 + 1
    max_year = max_year % 2006 + 1
    min_idx = 252*(min_year-1)
    max_idx = 252*max_year - 1
    val_idx = range(min_idx, max_idx)

    S, P, sigall = [x.to(device) for x in load_states(asset_type)]

    bp = torch.tensor(2e-3, device=device)
    tgt_vol = torch.tensor(0.1, device=device)

    # MACD baseline
    macds = S[:,:,-1,5]
    macd_actions_all = (macds*torch.exp(-macds**2/4))/0.89

    params = torch.load("models/dqn_index_1_5_best.pt")
    model.load_state_dict(params)
    model.eval()

    dqn_returns = []
    lo_returns = []
    macd_returns = []
    sgn_returns = []

    dqn_actions_prev = torch.zeros(len(S), device=device)
    sgn_actions_prev = torch.zeros(len(S), device=device)

    lo_actions = torch.ones(len(S), device=device)

    for t in val_idx:
        states = S[:, t]
        prices = P[:, t]
        prices_next = P[:, t+1]
        sigmas = sigall[:, t+1]
        sigmas_prev = sigall[:, t]

        dqn_actions = model.get_actions(states)
        dqn_rewards = get_reward(prices, prices_next, sigmas, sigmas_prev, dqn_actions, dqn_actions_prev, tgt_vol, bp)
        dqn_returns.append(dqn_rewards.mean().item())
        dqn_actions_prev = dqn_actions

        lo_rewards = get_reward(prices, prices_next, sigmas, sigmas_prev, lo_actions, lo_actions, tgt_vol, bp)
        lo_returns.append(lo_rewards.mean().item())

        macd_rewards = get_reward(prices, prices_next, sigmas, sigmas_prev, macd_actions_all[:, t],
                        macd_actions_all[:, t-1], tgt_vol, bp)
        macd_returns.append(macd_rewards.mean().item())

        sgn_actions = signR(P, t)
        sgn_rewards = get_reward(prices, prices_next, sigmas, sigmas_prev, sgn_actions, sgn_actions_prev, tgt_vol, bp)
        sgn_returns.append(sgn_rewards.mean().item())
        sgn_actions_prev = sgn_actions

    plt.plot(np.cumsum(dqn_returns), color='red')
    plt.plot(np.cumsum(lo_returns), color='blue')
    plt.plot(np.cumsum(macd_returns), color='green')
    plt.plot(np.cumsum(sgn_returns), color='orange')
    plt.legend(['DQN', 'Long-Only', 'MACD', 'Sign(R)'])

    plt.savefig('plots/dqn_index_2011_2015.png', format='png', bbox_inches='tight')

    plt.show()
    plt.clf()
