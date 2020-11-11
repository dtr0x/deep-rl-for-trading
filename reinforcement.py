import numpy as np
import pandas as pd
import torch

# pre-compute ex ante sigmas for price series
def ex_ante_sigma(prices, t0):
    r = prices[1:] - prices[:-1]
    ewma = pd.Series(r).ewm(span=60).std()[t0-2:]
    return np.array(ewma)

# pre-compute ex ante sigmas for all days/assets
def get_sigma_all(prices, t0):
    f = lambda x: ex_ante_sigma(x, t0) # function to compute sigma over price series
    sigall = np.apply_along_axis(f, 1, prices)
    return torch.tensor(sigall, dtype=torch.float32)

# Compute reward (Equation (4) in 'Deep Reinforcement Learning for Trading' with mu=1).
def get_reward(prices, prices_next, sigmas, sigmas_prev, actions, actions_prev, tgt_vol, bp):
    r = prices_next - prices
    x = actions*tgt_vol/sigmas
    y = actions_prev*tgt_vol/sigmas_prev
    return x*r - bp*prices*torch.abs(x-y)

if __name__ == '__main__':
    # pre-compute all ex ante sigma values
    data = pd.read_csv('cleaned_data.csv')
    # initial time index
    t0 = len(data)-252*5*3
    prices = np.array(data.drop(columns='date'), dtype='float32').transpose()
    sigall = get_sigma_all(prices, t0)
    torch.save(sigall, 'ex_ante_sigma.pt')
