import torch
import numpy as np
import pandas as pd

# convert 1D dataframe to tensor (avoid inporting tensorflow)
def df2tensor(df):
    return torch.tensor(np.array(df))

# normalize data (tensor)
def normalize(data):
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data-mu)/sigma

# normalized returns from price data up to time t
def normalize_returns(data, t):
    return1Month = (data[t]-data[t-30])/data[t-30]
    return2Month = (data[t]-data[t-60])/data[t-60]
    return3Month = (data[t]-data[t-90])/data[t-90]
    return1Year = (data[t]-data[t-252])/data[t-252]

    totalReturn = (data[1:(t+1)]-data[0:t])/data[0:t]
    df = pd.DataFrame(totalReturn)

    # the first value of each tensor is nan here. Verify how df.ewn works
    EWSTD1Month = df2tensor(df.ewm(span=5).std())
    EWSTD2Month = df2tensor(df.ewm(span=10).std())
    EWSTD3Month = df2tensor(df.ewm(span=15).std())
    EWSTD1Year = df2tensor(df.ewm(span=60).std())

    # don't need [[]] to index
    sd1Month = EWSTD1Month[t-1]
    sd2Month = EWSTD2Month[t-1]
    sd3Month = EWSTD3Month[t-1]
    sd1Year = EWSTD1Year[t-1]

    # use np.sqrt to avoid importing math module
    finalReturn1Month = return1Month/(sd1Month*np.sqrt(30))
    finalReturn2Month = return2Month/(sd2Month*np.sqrt(60))
    finalReturn3Month = return3Month/(sd3Month*np.sqrt(90))
    finalReturn1Year = return1Year/(sd1Year*np.sqrt(252))

    normalReturns = [finalReturn1Month,finalReturn2Month,finalReturn3Month,finalReturn1Year]
    return df2tensor(normalReturns)

def MACD(data, t):
    StdRoll63days  = torch.std(data[(t-63):(t+1)])
    StdRoll252days = torch.std(data[(t-252):(t+1)])

    df = pd.DataFrame(data[:(t+1)])

    # define these and convert to tensor all at once
    EWMA1short = df2tensor(df.ewm(span=8).mean())
    EWMA2short = df2tensor(df.ewm(span=16).mean())
    EWMA3short = df2tensor(df.ewm(span=32).mean())
    EWMA1long  = df2tensor(df.ewm(span=24).mean())
    EWMA2long  = df2tensor(df.ewm(span=48).mean())
    EWMA3long  = df2tensor(df.ewm(span=96).mean())

    maShort = [EWMA1short[t], EWMA2short[t], EWMA3short[t]]
    maLong = [EWMA1long[t], EWMA2long[t], EWMA3long[t]]

    # use a loop for q instead of hard-coding indices
    q = []
    for i in range(3):
      ms = maShort[i]
      for j in range(3):
          ml = maLong[j]
          q.append((ms-ml)/StdRoll63days)

    # don't use the tensorflow divide function for this, it's scalar division
    macdVec = torch.tensor(q)/StdRoll252days

    # no need to hard code mean using indices
    return macdVec.mean()

if __name__ == '__main__':
    # load data
    SP500 = pd.read_csv('SP500 dec2004-dec2018.csv')

    # get normalized close prices
    close_prices = torch.tensor(SP500['Close'])
    norm_prices = normalize(close_prices)

    # print with default formatting
    print("Normalized close prices: {}".format(norm_prices))
    print("Number of days: {}".format(norm_prices.size()[0]))

    t = 252
    norm_returns = normalize_returns(close_prices, t)
    print("Normalized returns at time t={}: {}".format(t, norm_returns))

    tensor_list = [torch.tensor(-0.7582, dtype=torch.float64),
                 torch.tensor(-0.7849, dtype=torch.float64),
                 torch.tensor(-0.7931, dtype=torch.float64),
                 torch.tensor(-0.7852, dtype=torch.float64)]
    l2t = torch.tensor(tensor_list)
    print("Test list to tensor: {}".format(l2t))

    macd = MACD(close_prices, t)
    # {:.3f} is a placeholder to convert a float to 3 decimal places
    print("MACD: {:.3f}".format(macd))
