from util import *

# Functions for baseline algorithms

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

def MACD1(data,t):

    df1 = pd.DataFrame(data[(t-251):(t+1)])
    df2 = data[(t-62):(t+1)]

    StdRoll63days = df2.std()

    EWMA1short = df1.ewm(span=8).mean()
    EWMA2short = df1.ewm(span=16).mean()
    EWMA3short = df1.ewm(span=32).mean()
    EWMA1long  = df1.ewm(span=24).mean()
    EWMA2long  = df1.ewm(span=48).mean()
    EWMA3long  = df1.ewm(span=96).mean()

    maShort = EWMA1short, EWMA2short, EWMA3short
    maLong = EWMA1long, EWMA2long, EWMA3long

    q = [[0]]*3
    StdRoll252daystest = [0]*3
    macdVec = [0]*3
    for i in range(3):
      q[i] = df2tensor(maShort[i] - maLong[i])/StdRoll63days
      StdRoll252daystest[i] = q[i].std()
      macdVec[i] = df2tensor(q[i])[-1]/StdRoll252daystest[i]

    return df2tensor(macdVec).mean()

# MACD SIGNAL
def MACD_signal(data,t):

    MACD_value = MACD(data,t)
    MACD_signal = (MACD_value*torch.exp(-MACD_value**2/4))/0.89

    return MACD_signal

def MACD_reward(data,t):

    action_vec = df2tensor([MACD_signal(data,t),MACD_signal(data,t-1)])

    r_vec = data[(t-251):(t+1)]-data[(t-252):t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-1],ex_ante_sigma[-2]])

    r = data[t+1] - data[t]
    p = data[t]

    tgt_volatility = 0.15
    bp = 0.002

    reward1 = action_vec[0]*r*tgt_volatility/sigma_vec[0]
    reward2 = action_vec[0]*tgt_volatility/sigma_vec[0] - action_vec[1]*tgt_volatility/sigma_vec[1]
    reward  = reward1 - bp*p*abs(reward2)

    return reward



def long_only_reward(data,t):

    action_vec = df2tensor([1,1])
    r_vec = data[(t-251):(t+1)]-data[(t-252):t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-1],ex_ante_sigma[-2]])

    r = data[t+1] - data[t]
    p = data[t]

    tgt_volatility = 0.15
    bp = 0.002

    reward1 = action_vec[0]*r*tgt_volatility/sigma_vec[0]
    reward2 = action_vec[0]*tgt_volatility/sigma_vec[0] - action_vec[1]*tgt_volatility/sigma_vec[1]
    reward  = reward1 - bp*p*abs(reward2)

    return reward



def sign_reward(data,t):

    #expected_trend1 = torch.true_divide(data[t]-data[t-252],data[t-252])
    #expected_trend2 = torch.true_divide(data[t-1]-data[t-253],data[t-253])

    data_new = pd.DataFrame(data)

    data1 = data_new[(t-251):(t+1)]
    data2 = data_new[(t-252):t]
    data3 = data_new[(t-253):(t-1)]
    data1 = data1.reset_index()[0]
    data2 = data2.reset_index()[0]
    data3 = data3.reset_index()[0]

    df1 = df2tensor((data1-data2)/data2)
    df2 = df2tensor((data2-data3)/data3)

    expected_trend1 = df1.mean()
    expected_trend2 = df2.mean()

    #taking a maximum long position when the expected trend is positive
    action_vec = [1]*2
    if expected_trend1<0:
       action_vec[0] = -1
    if expected_trend2<0:
       action_vec[1] = -1

    r_vec = data[(t-251):(t+1)]-data[(t-252):t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-1],ex_ante_sigma[-2]])

    r = data[t+1] - data[t]
    p = data[t]

    tgt_volatility = 0.15
    bp = 0.002

    reward1 = action_vec[0]*r*tgt_volatility/sigma_vec[0]
    reward2 = action_vec[0]*tgt_volatility/sigma_vec[0] - action_vec[1]*tgt_volatility/sigma_vec[1]
    reward  = reward1 - bp*p*abs(reward2)

    return reward
