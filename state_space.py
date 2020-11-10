import numpy as np
import pandas as pd

'''Helper functions to compute state space from price data.'''

# normalized prices for last 60 days up to time t
def norm_prices(prices, t):
    x = prices[:t+1]
    mu = np.mean(x)
    sigma = np.std(x)
    return ((x-mu)/sigma)[-60:]

# normalized returns for last 60 days up to time t
def norm_returns(prices, t):
    periods = [21,42,63,252]
    rets = [(prices[p:t+1]-prices[:t-p+1])[-60:] for p in periods]
    rets_all = prices[1:t+1] - prices[:t]
    ewma = np.array(pd.Series(rets_all).ewm(span=60).std())[-60:]
    normalReturns = [r/ewma/np.sqrt(p) for r, p in zip(rets, periods)]
    return np.stack(normalReturns)

# MACDs for last 60 days up to time t
def macd(prices, t):
    x = prices[:t+1]
    s = pd.Series(x)
    stdRoll63Days = np.array(s.rolling(63).std())
    maShort = [np.array(s.ewm(span=i).mean()) for i in [8, 16, 32]]
    maLong = [np.array(s.ewm(span=i).mean()) for i in [24, 48, 96]]
    q = [(mS - mL)/stdRoll63Days for mS, mL in zip(maShort, maLong)]
    q_std = [np.array(pd.Series(q_i).rolling(252).std()) for q_i in q]
    macds = np.stack([x/y for x,y in zip(q, q_std)])[:, -60:]
    return macds.mean(axis=0)

# RSI for last 60 days up to time t
def rsi(prices, t):
    x = prices[:t+1]
    diff  = x[1:] - x[:-1]
    up_chg = np.zeros_like(diff)
    down_chg = np.zeros_like(diff)
    # up change is equal to the positive difference, otherwise equal to zero
    u_idx = np.where(diff > 0)[0]
    up_chg[u_idx] = diff[u_idx]
    # down change is equal to negative difference, otherwise equal to zero
    d_idx = np.where(diff < 0)[0]
    down_chg[d_idx] = np.abs(diff[d_idx])
    # smoothing factor
    alpha = 1/30
    up_chg_avg = np.array(pd.Series(up_chg).ewm(alpha=alpha).mean())[-60:]
    down_chg_avg = np.array(pd.Series(down_chg).ewm(alpha=alpha).mean())[-60:]
    # compute RSI
    rs = up_chg_avg/down_chg_avg
    rsi = 100 - 100/(1+rs)
    return rsi

# Get state for time t, dim=60x7
def daily_state(prices, t):
    state = np.zeros((7, 60), dtype='float32')
    state[0] = norm_prices(prices, t)
    state[1:5] = norm_returns(prices, t)
    state[5] = macd(prices, t)
    state[6] = rsi(prices, t)
    return state.transpose()

# Compute all states from price series, starting from an initial time
def get_states_iter(prices, t0):
    n_states = len(prices) - t0
    states = np.zeros((n_states, 60, 7), dtype='float32')
    i = 0
    for t in range(t0, len(prices)):
        states[i] = daily_state(prices, t)
        i += 1
    return states
