import torch
import numpy as np
import pandas as pd

# convert 1D dataframe to tensor (avoid inporting tensorflow)
def df2tensor(df):
    return torch.tensor(np.array(df))

# normalize data (tensor)
def normalize_prices(data):
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data-mu)/sigma

# normalized returns from price data up to time t
def normalize_returns(data, t):
    
    return1Month = (data[t]-data[t-30])/data[t-30]
    return2Month = (data[t]-data[t-60])/data[t-60]
    return3Month = (data[t]-data[t-90])/data[t-90]
    return1Year    = (data[t]-data[t-252])/data[t-252]

    totalReturn = (data[1:(t+1)]-data[0:t])/data[0:t]

    df = pd.DataFrame(totalReturn)

    # the first value of each tensor is nan here. Verify how df.ewn works
    EWSTD1Month = df2tensor(df.ewm(span=5).std())
    EWSTD2Month = df2tensor(df.ewm(span=10).std())
    EWSTD3Month = df2tensor(df.ewm(span=15).std())
    EWSTD1Year  = df2tensor(df.ewm(span=60).std())

    # don't need [[]] to index
    sd1Month = EWSTD1Month[-1]
    sd2Month = EWSTD2Month[-1]
    sd3Month = EWSTD3Month[-1]
    sd1Year  = EWSTD1Year[-1]

    # use np.sqrt to avoid importing math module
    finalReturn1Month = return1Month/(sd1Month*np.sqrt(30))
    finalReturn2Month = return2Month/(sd2Month*np.sqrt(60))
    finalReturn3Month = return3Month/(sd3Month*np.sqrt(90))
    finalReturn1Year  = return1Year/(sd1Year*np.sqrt(252))

    normalReturns = [finalReturn1Month,finalReturn2Month,finalReturn3Month,finalReturn1Year]

    return df2tensor(normalReturns)

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
    
    

def RSI (data,t):

    # 0:t+1 because index returns back [0:t+1) not inclusive!
    df    = pd.DataFrame(data[0:(t+1)])
    diff  = df.diff(1).dropna()

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    # we set com=time_window-1 so we get decay alpha=1/time_window
    # where time_window = 30 (from paper)
    time_window = 30
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)

    return df2tensor(rsi)[-1]

# Function that returns state space for one asset at time t i.e. a vector of size 7
def state_space1 (data,t):

    normalized_price  = normalize_prices(data)[t]
    normalized_return = normalize_returns(data,t)
    MACD_value        = MACD(data,t)
    RSI_value         = RSI(data,t)

    one_mth_return    = normalized_return[0]
    two_mth_return    = normalized_return[1]
    three_mth_return  = normalized_return[2]
    one_yr_return     = normalized_return[3]

    state_space = [normalized_price,
                   one_mth_return,
                   two_mth_return,
                   three_mth_return,
                   one_yr_return,
                   MACD_value,
                   RSI_value]

    return df2tensor(state_space)


# Function that returns state space for one asset at time t to t-60 i.e. dimension 7*60
def state_space2 (data,t):

    state_space_info = [[]]*60
    for i in range(60) :
        tt = t-i
        state_space_info[i] = np.array(state_space1(data,tt))

    state_space_info = np.array(state_space_info)
    state_space      = [list(x) for x in state_space_info.transpose()]

    return df2tensor(state_space)
    
    
# Function that returns state space for one asset at time t to t-60 for days t i.e. dimension t*7*60
def state_space3 (data):

    t = 312
    T = list(data.shape)[0]-1
    t_days = len(range(t,T))

    # initialize the tensor of desired size with 0s
    state_space_tvec = torch.zeros(size=(t_days+1, 7, 60))
    for j in range(t_days):
      state_space_tvec[j, :, :] = state_space2(data,t)

    return state_space_tvec
    
    
# Function that returns state space by asset dimension n*7*60*t for all days t i.e. dimension nbr specific assets * t * 7 * 60
# asset is a character: asset_type = "Comdty", "Curncy" or "Index"
def state_space4 (data,asset_type):

    data = data.loc[:,data.columns.str.endswith(asset_type)]
    nbr_assets = data.shape[1]
    data.columns = range(0,nbr_assets)

    t = 312
    T = list(data.shape)[0]-1
    t_days = len(range(t,T))
    state_space_n_t = torch.zeros(size=(nbr_assets, t_days, 7, 60))

    for i in range(nbr_assets):
        df = torch.tensor(data[i])
        state_space_n_t[i,:, :, :] = state_space3(df)

return state_space_n_t


# Function that returns state space by asset dimension n*7*60*t for all days t i.e. dimension nbr. Total assets * t * 7 * 60
def state_space5 (data):

    nbr_assets = data.shape[1]
    data.columns = range(0,nbr_assets)

    t = 312
    T = list(data.shape)[0]-1
    t_days = len(range(t,T))
    state_space_n_t = torch.zeros(size=(nbr_assets, t_days, 7, 60))

    for i in range(nbr_assets):
      df = torch.tensor(data[i])
      state_space_n_t[i,:, :, :] = state_space3(df)

    return state_space_n_t






if __name__ == '__main__':
    # load data
    SP500 = pd.read_csv('SP500 dec2004-dec2018.csv')

    # get normalized closing prices
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
