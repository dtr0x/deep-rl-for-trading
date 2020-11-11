import torch
from policy_net import PolicyNet
from save_states_all import load_states
from reinforcement import get_reward
import numpy as np
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def signR(prices, t):
    rets = (prices[:, t-251:t+1] - prices[:, t-252:t])/prices[:, t-252:t]
    return rets.mean(axis=1).sign()

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

    # load params and set model to eval mode
    params1 = torch.load("models/dqn_{}_{}_{}.pt".format(asset_type, 2006, 2010))
    params2 = torch.load("models/dqn_{}_{}_{}.pt".format(asset_type, 2011, 2015))
    model = PolicyNet().to(device).eval()

    min_year_idx = min_year % 2006 + 1
    max_year_idx = max_year % 2006 + 1
    min_idx = 252*(min_year_idx-1)
    max_idx = 252*max_year_idx - 1
    test_idx = range(min_idx, max_idx)

    # index to switch params
    switch_idx = 252*(2016 % 2006)

    S, P, sigall = [x.to(device) for x in load_states(asset_type)]

    bp = torch.tensor(2e-3, device=device)
    tgt_vol = torch.tensor(0.1, device=device)

    dqn_returns = []
    lo_returns = []
    macd_returns = []
    sgn_returns = []

    dqn_actions_prev = torch.zeros(len(S), device=device)
    sgn_actions_prev = torch.zeros(len(S), device=device)

    # long only actions
    lo_actions = torch.ones(len(S), device=device)

    # MACD actions
    macds = S[:,:,-1,5]
    macd_actions = (macds*torch.exp(-macds**2/4))/0.89

    # set initial model params
    if min_idx < switch_idx:
        model.load_state_dict(params1)
    else:
        model.load_state_dict(params2)

    for t in test_idx:
        # change the model if we reach switch_idx
        if t == switch_idx:
            model.load_state_dict(params2)

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

        macd_rewards = get_reward(prices, prices_next, sigmas, sigmas_prev, macd_actions[:, t],
                        macd_actions[:, t-1], tgt_vol, bp)
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

    if asset_type == 'all':
        asset_title = 'all assets'
    elif asset_type == 'commodity':
        asset_title = 'commodities'
    elif asset_type == 'fixed_income':
        asset_title = 'fixed income'
    elif asset_type == 'currency':
        asset_title = 'currencies'
    elif asset_type == 'index':
        asset_title = 'equities'
    plt.title("Cumulative daily returns ({})".format(asset_title))

    x_labels = [str(y) for y in list(np.arange(min_year, max_year+1))]
    tick_idx = np.arange(0, (max_year-min_year)*252+1, 252)
    plt.xticks(tick_idx, x_labels)

    plt.savefig("plots/dqn_{}_{}_{}.png".format(asset_type, min_year, max_year),
    format='png', bbox_inches='tight')

    plt.show()
    plt.clf()
