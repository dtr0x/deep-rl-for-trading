import numpy as np
import pandas as pd
import torch

# pre-compute ex ante sigmas for price series
def ex_ante_sigma(prices, t0):
    n_days = len(prices) - t0
    sigmas = np.zeros((n_days, 2), dtype='float32')
    i = 0
    for t in range(t0, len(prices)):
        r = prices[(t-251):(t+1)] - prices[(t-252):t]
        s = pd.Series(r).ewm(span=60).std().tolist()[-2:]
        sigmas[i] = s
        i += 1
    return sigmas

# pre-compute ex ante sigmas for all days/assets
def get_sigma_all(prices, t0):
    n_assets = len(prices)
    f = lambda x: ex_ante_sigma(x, t0) # function to compute sigma over price series
    sigall = np.apply_along_axis(f, 1, prices)
    return torch.tensor(sigall, dtype=torch.float32)

# Compute reward (Equation (4) in 'Deep Reinforcement Learning for Trading' with mu=1).
def get_reward(prices, prices_next, sigma, actions, actions_prev, tgt_vol, bp):
    r = prices_next - prices
    x = actions*tgt_vol/sigma[:,1]
    y = actions_prev*tgt_vol/sigma[:,0]
    return x*r - bp*prices*torch.abs(x-y)

if __name__ == '__main__':
    # pre-compute all ex ante sigma values
    data = pd.read_csv('cleaned_data.csv')
    # initial time index
    t0 = len(data)-252*5*3
    prices = np.array(data.drop(columns='date'), dtype='float32').transpose()
    sigall = get_sigma_all(prices, t0)
    torch.save(sigall, 'ex_ante_sigma.pt')
