from state_space import get_states_iter
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch, time, re

'''Compute all states for all assets in parallel and save data.'''

# Helper function to load all states, corresponding prices and ex ante sigma values
# accepted asset_type: 'all', 'commodity', 'currency', 'index'
def load_states(asset_type='all'):
    state_data = np.load('state_data.npz')

    S = torch.tensor(state_data['states'], dtype=torch.float32)
    P =  torch.tensor(state_data['prices'].transpose(), dtype=torch.float32)

    if asset_type == 'all':
        return S, P
    else:
        if asset_type == 'commodity':
            r = r'Comdty$'
        elif asset_type == 'currency':
            r = r'Curncy$'
        elif asset_type == 'index':
            r = r'Index$'
        else:
            return None

        cols = pd.read_csv('cleaned_data.csv').columns[1:]
        # get indices for assets of specified type
        asset_idx = np.where([re.search(r, c) for c in cols])[0]

        S = S[asset_idx] # n_assets * n_days * 60 * 7
        P = P[asset_idx] # n_assets * n_days
        # ex ante sigma values for each day to compute rewards
        sigall = torch.load('ex_ante_sigma.pt')[asset_idx] # n_assets * n_days * 2

        return S, P, sigall

if __name__ == '__main__':
    data = pd.read_csv('cleaned_data.csv')
    # initial time index: compute and save states starting from 2005
    t0 = data[data['date'] >= '2005'].index[0]
    prices = np.array(data.drop(columns='date'), dtype='float32')

    # initialize processing pool
    n_cpus = mp.cpu_count()
    print("Number of CPUs: {}".format(n_cpus))
    pool = mp.Pool(n_cpus)

    t1 = time.time()
    # compute states for all assets in parallel (prices row-wise)
    result = [pool.apply_async(get_states_iter, args=(x, t0)) for x in prices.transpose()]
    states_all = np.array([r.get() for r in result])
    t2 = time.time()
    print("Computed states for all assets in {:.2f} minutes.".format((t2-t1)/60))

    # save states and corresponding prices (compressed)
    np.savez_compressed('state_data.npz', prices=prices[t0:], states=states_all)
