from baselines import *

# INPUT: raw data, time t, action at time t-1 and action at time t
# OUTPUT: reward at time t+1
def reward(data,t,action_prev,action_now):

    action_vec = [0]*2
    action_vec[0] = action_prev
    action_vec[1] = action_now

    r_vec = data[(t-251):(t+1)]-data[(t-252):t]
    r_df  = pd.DataFrame(r_vec)
    ex_ante_sigma = df2tensor(r_df.ewm(span=60).std())

    sigma_vec = df2tensor([ex_ante_sigma[-1],ex_ante_sigma[-2]])

    r = data[t+1] - data[t]
    p = data[t]

    tgt_volatility = 0.10
    bp = 0.002

    reward1 = action_vec[0]*r*tgt_volatility/sigma_vec[0]
    reward2 = action_vec[0]*tgt_volatility/sigma_vec[0] - action_vec[1]*tgt_volatility/sigma_vec[1]
    reward  = reward1 - bp*p*abs(reward2)

    return reward

# INPUT: data set,  time t, tensor of action at time t-1, and
#        tensor of actions at time t
# OUTPUT: tensor of rewards for time t+1
def get_reward(data,t,action_vec_prev,action_vec_now):

    nbr_assets = len(action_vec_prev)
    data.columns = range(0,nbr_assets)

    reward_vec = [0]*nbr_assets
    for i in range(nbr_assets):
        df = df2tensor(data[i])
        reward_vec[i] = reward(df,t,action_vec_prev[i],action_vec_now[i])

    return df2tensor(reward_vec)
