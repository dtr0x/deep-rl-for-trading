from state_space import get_states_iter
import numpy as np
import pandas as pd
import multiprocessing as mp
import time, re

'''Compute all states for all assets in parallel and save data.'''

# Helper function to load all states and corresponding prices
# accepted asset_type: 'all', 'commodity', 'currency', 'index'
def load_states(asset_type='all'):
    state_data = np.load('state_data.npz')

    P = state_data['prices']
    S = state_data['states']

    if asset_type == 'all':
        return P, S
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

        P = P[:, asset_idx] # n_days * n_assets
        S = S[asset_idx] # n_assets * n_days * 60 * 7

        return P, S

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
    result = [pool.apply_async(get_states_iter, args=(x, t0)) for x in prices.transpose()[:, :(t0+20)]]
    states_all = np.array([r.get() for r in result])
    t2 = time.time()
    print("Computed states for all assets in {:.2f} minutes.".format((t2-t1)/60))

    # save states and corresponding prices (compressed)
    np.savez_compressed('state_data.npz', prices=prices[t0:(t0+20), :], states=states_all)
