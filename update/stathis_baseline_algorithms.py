
import torch
import numpy as np
import pandas as pd

SP500       = pd.read_csv('/content/SP500 dec2004-dec2018.csv')
StathisData = pd.read_csv('/content/Final Stathis Data (1).csv')

SP_prices = SP500.loc[:,'Close']
SPprices = torch.tensor(SP_prices) 
data = SPprices
t    = 312

# convert 1D dataframe to tensor (avoid inporting tensorflow)
def df2tensor(df):
    return torch.tensor(np.array(df))

def MACD(data,t):

    df = pd.DataFrame(data[:(t+1)])
    StdRoll63days = df2tensor(df.rolling(63).std())[-1]

    EWMA1short = df.ewm(span=8).mean()
    EWMA2short = df.ewm(span=16).mean()
    EWMA3short = df.ewm(span=32).mean()
    EWMA1long  = df.ewm(span=24).mean()
    EWMA2long  = df.ewm(span=48).mean()
    EWMA3long  = df.ewm(span=96).mean()

    maShort = EWMA1short, EWMA2short, EWMA3short
    maLong = EWMA1long, EWMA2long, EWMA3long

    q = [[0]]*3
    StdRoll252daystest = [0]*3
    macdVec = [0]*3
    for i in range(3):
      q[i] = (maShort[i] - maLong[i])/StdRoll63days
      StdRoll252daystest[i] = df2tensor(q[i].rolling(252).std())[-1]
      macdVec[i] = df2tensor(q[i])[-1]/StdRoll252daystest[i]

    return df2tensor(macdVec).mean()


# MACD SIGNAL
def MACD_signal(data,t):

    MACD_value = MACD(data,t)
    MACD_signal = (MACD_value*torch.exp(-MACD_value**2/4))/0.89
                   
    return MACD_signal

def MACD_reward(data,t):

    action_vec = df2tensor([MACD_signal(data,t-1),MACD_signal(data,t-2)])

    r_vec = data[1:(t+1)] - data[0:t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-2],ex_ante_sigma[-3]])

    r = r_vec[-1]
    p = data[t]

    #tgt_volatility = ex_ante_sigma[1,-1].mean()
    #tgt_volatility = ex_ante_sigma[1,-1].median()
    #tgt_volatility = 0.15
    tgt_volatility = 0.40
    #tgt_volatility = 0.10
    #tgt_volatility = (15+40)/2

    bp = 0.0001
    #bp = 0.001
    #bp = 0.01
    #bp = 0.1
    #bp = 1

    reward1 = action_vec[0]*tgt_volatility/sigma_vec[0]*r
    reward2 = tgt_volatility/sigma_vec[0]*action_vec[0] - tgt_volatility/sigma_vec[1]*action_vec[1]
    reward = reward1 - bp*p*abs(reward2)

    return reward


def long_only_reward(data,t):

    action_vec = df2tensor([1,1])

    r_vec = data[1:(t+1)] - data[0:t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-2],ex_ante_sigma[-3]])

    r = r_vec[-1]
    p = data[t]

    #tgt_volatility = ex_ante_sigma[1,-1].mean()
    #tgt_volatility = ex_ante_sigma[1,-1].median()
    #tgt_volatility = 0.15
    tgt_volatility = 0.40
    #tgt_volatility = 0.10
    #tgt_volatility = (15+40)/2

    bp = 0.0001
    #bp = 0.001
    #bp = 0.01
    #bp = 0.1
    #bp = 1

    reward1 = action_vec[0]*tgt_volatility/sigma_vec[0]*r
    reward2 = tgt_volatility/sigma_vec[0]*action_vec[0] - tgt_volatility/sigma_vec[1]*action_vec[1]
    reward = reward1 - bp*p*abs(reward2)

    return reward


def Sgn_reward(data,t):

    #df1 = (data[(t-251):(t+1)]-data[(t-252):t])/data[(t-252):t]
    #df2 = (data[(t-252):t]-data[(t-253):(t-1)])/data[(t-253):(t-1)]

    expected_trend1 = (data[t]-data[t-252])/data[t-252]
    expected_trend2 = (data[t-1] - data[t-253])/data[t-253]

    #expected_trend1 = df1.mean()
    #expected_trend2 = df2.mean()

    #taking a maximum long position when the expected trend is positive
    action_vec = [1]*2
    if expected_trend1<0:
      action_vec[0] = -1
    if expected_trend2<0:
      action_vec[1] = -1

    r_vec = data[1:(t+1)] - data[0:t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-2],ex_ante_sigma[-3]])

    r = r_vec[-1]
    p = data[t]

    #tgt_volatility = ex_ante_sigma[1,-1].mean()
    #tgt_volatility = ex_ante_sigma[1,-1].median()
    #tgt_volatility = 0.15
    tgt_volatility = 0.40
    #tgt_volatility = (15+40)/2

    bp = 0.0001
    #bp = 0.001
    #bp = 0.01
    #bp = 0.1
    #bp = 1

    reward1 = action_vec[0]*tgt_volatility/sigma_vec[0]*r
    reward2 = tgt_volatility/sigma_vec[0]*action_vec[0] - tgt_volatility/sigma_vec[1]*action_vec[1]
    reward = reward1 - bp*p*abs(reward2)

    return reward



df = StathisData
df = df.loc[:,df.columns.str.endswith("Comdty")]
nbr_assets = df.shape[1]
df.columns = range(0,nbr_assets)
t = 3405
T = 3647
TT = len(range(t,T+1))
N = df.shape[1]
MACD_reward_vec  = [0]*N
MACD_reward_vect = [0]*TT
for tt in range(t,T+1):
  for i in range(N):
    df1 = df2tensor(df[i][0:(tt+1)])
    MACD_reward_vec[i] = MACD_reward(df1,tt)
  MACD_reward_vect[tt-t] = df2tensor(MACD_reward_vec).mean()

long_only_reward_vec  = [0]*N
long_only_reward_vect = [0]*TT
for tt in range(t,T+1):
  for i in range(N):
    df1 = df2tensor(df[i][0:(tt+1)])
    long_only_reward_vec[i] = long_only_reward(df1,tt)
  long_only_reward_vect[tt-t] = df2tensor(long_only_reward_vec).mean()

Sgn_reward_vec  = [0]*N
Sgn_reward_vect = [0]*TT
for tt in range(t,T+1):
  for i in range(N):
    df1 = df2tensor(df[i][0:(tt+1)])
    Sgn_reward_vec[i] = Sgn_reward(df1,tt)
  Sgn_reward_vect[tt-t] = df2tensor(Sgn_reward_vec).mean()


# TESTING FUNCTION

# MEAN
df2tensor(MACD_reward_vect).mean()
df2tensor(long_only_reward_vect).mean()
df2tensor(Sgn_reward_vect).mean()

# STANDARD DEVIATION
df2tensor(MACD_reward_vect).std()
df2tensor(long_only_reward_vect).std()
df2tensor(Sgn_reward_vect).std()


