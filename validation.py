import torch
from policy_net import PolicyNet
from save_states_all import load_states
from reinforcement import reward
import numpy as np
import matplotlib.pyplot as plt

def signR(prices, t):
    rets = (prices[:, t-251:t+1] - prices[:, t-252:t])/prices[:, t-252:t]
    return rets.mean(axis=1).sign()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyNet().to(device)

    S, P, sigall = [x.to(device) for x in load_states('commodity')]
    nidx = [7,16,20,25,27,32,35]
    idx = [i for i in np.arange(len(S)) if i not in nidx]
    S = S[idx]
    P = P[idx]
    sigall = sigall[idx]

    val_idx = range(2500,4000)

    bp = torch.tensor(2e-3, device=device)
    tgt_vol = torch.tensor(0.1, device=device)

    # MACD baseline
    macds = S[:,:,-1,5]
    macd_actions_all = (macds*torch.exp(-macds**2/4))/0.89

    # long actions
    lo_actions_all = torch.ones_like(macd_actions_all)

    params = torch.load("models/pg_epoch_20.pt")
    model.load_state_dict(params)
    model.eval()

    dqn_returns = []
    lo_returns = []
    macd_returns = []
    sgn_returns = []

    dqn_actions_prev = torch.zeros(len(S), device=device)
    lo_actions_prev = torch.zeros(len(S), device=device)
    macd_actions_prev = torch.zeros(len(S), device=device)
    sgn_actions_prev = torch.zeros(len(S), device=device)
    for t in val_idx:
        states = S[:, t]
        next_states = S[:, t+1]
        prices = P[:, t]
        prices_next = P[:, t+1]
        sigmas = sigall[:, t]

        dqn_actions = model.get_actions(states)
        dqn_rewards = reward(prices, prices_next, sigmas, dqn_actions, dqn_actions_prev, tgt_vol, bp)
        dqn_returns.append(dqn_rewards.mean().item())

        lo_actions = lo_actions_all[:, t]
        lo_rewards = reward(prices, prices_next, sigmas, lo_actions, lo_actions_prev, tgt_vol, bp)
        lo_returns.append(lo_rewards.mean().item())

        macd_actions = macd_actions_all[:, t]
        macd_rewards = reward(prices, prices_next, sigmas, macd_actions, macd_actions_prev, tgt_vol, bp)
        macd_returns.append(macd_rewards.mean().item())

        sgn_actions = signR(P, t)
        sgn_rewards = reward(prices, prices_next, sigmas, sgn_actions, sgn_actions_prev, tgt_vol, bp)
        sgn_returns.append(sgn_rewards.mean().item())

        dqn_actions_prev = dqn_actions
        lo_actions_prev = lo_actions
        macd_actions_prev = macd_actions
        sgn_actions_prev = sgn_actions

    plt.plot(np.cumsum(dqn_returns), color='red')
    plt.plot(np.cumsum(lo_returns), color='blue')
    plt.plot(np.cumsum(macd_returns), color='green')
    plt.plot(np.cumsum(sgn_returns), color='orange')
    plt.legend(['PG', 'Long-Only', 'MACD', 'Sign(R)'])

    plt.savefig('plots/pg_comm_1.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
