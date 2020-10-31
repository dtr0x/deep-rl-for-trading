import numpy as np
import pandas as pd

'''Helper functions to compute state space from price data.'''

# normalized prices over one year up to time t, return last value
def norm_price(prices, t):
    x = prices[(t-251):(t+1)] # one year of data
    mu    = np.mean(x)
    sigma = np.std(x)
    return ((x-mu)/sigma)[-1]

# normalized returns over one year from price series up to time t
def norm_returns(prices, t):
    periods = [21,42,63,252]
    rets = [(prices[t]-prices[t-p])/prices[t-p] for p in periods]
    returns1Year = (prices[(t-251):(t+1)]-prices[(t-252):t])/prices[(t-252):t]
    s = pd.Series(returns1Year)
    stddev = [s.ewm(span=i).std().tolist()[-1] for i in [5,10,15,60]]
    normalReturns = [r/sd/np.sqrt(p) for r, sd, p in zip(rets, stddev, periods)]
    return np.array(normalReturns)

# MACD using one year look back window up to time t
def macd(prices, t):
    s = pd.Series(prices[(t-251):(t+1)])
    stdRoll63Days = s.rolling(63).std().tolist()[-1]
    maShort = np.array([s.ewm(span=i).mean().tolist() for i in [8, 16, 32]])
    maLong = np.array([s.ewm(span=i).mean().tolist() for i in [24, 48, 96]])
    q = ((maShort - maLong)/stdRoll63Days).transpose()
    stdRoll252Days = np.array(pd.DataFrame(q).rolling(252).std().tail(1))
    macd = q[-1, :]/stdRoll252Days
    return macd.mean()

# RSI indicator using one year look back window up to time t
def rsi(prices, t):
    x = prices[(t-251):(t+1)]
    diff  = x[1:] - x[:-1]
    up_chg = np.zeros_like(diff)
    down_chg = np.zeros_like(diff)
    # up change is equal to the positive difference, otherwise equal to zero
    u_idx = np.where(diff > 0)[0]
    up_chg[u_idx] = diff[u_idx]
    # down change is equal to negative deifference, otherwise equal to zero
    d_idx = np.where(diff < 0)[0]
    down_chg[d_idx] = diff[d_idx]
    # we set com=time_window-1 so we get decay alpha=1/time_window
    time_window = 30
    up_chg_avg = pd.Series(up_chg).ewm(com=time_window-1 , min_periods=time_window).mean().tolist()[-1]
    down_chg_avg = pd.Series(down_chg).ewm(com=time_window-1 , min_periods=time_window).mean().tolist()[-1]
    # compute RSI
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

# Function to retrieve a tensor of size 7 at time t
def daily_features(prices, t):
    f = np.zeros(7, dtype='float32')
    f[0] = norm_price(prices, t)
    f[1:5] = norm_returns(prices, t)
    f[5] = macd(prices, t)
    f[6] = rsi(prices, t)
    return f

# Get state for time t, dim=60x7
def daily_state(prices, t):
    state = []
    for i in range(t-59, t+1):
        state.append(daily_features(prices, i))
    return np.array(state)

# Compute all states from price series, starting from an initial time
def get_states_iter(prices, t0):
    n_states = len(prices) - t0
    states = np.zeros((n_states, 60, 7), dtype='float32')
    i = 0
    for t in range(t0, len(prices)):
        states[i] = daily_state(prices, t)
        i += 1
    return states
