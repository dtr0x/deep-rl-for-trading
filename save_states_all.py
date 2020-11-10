from state_space import get_states_iter
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch, time

'''Compute all states for all assets in parallel and save data.'''

# Helper function to load all states, corresponding prices and ex ante sigma values
# accepted asset_type: 'all', 'commodity', 'fixed_income', 'currency', 'index'
def load_states(asset_type='all'):
    state_data = np.load('state_data.npz')

    S = torch.tensor(state_data['states'], dtype=torch.float32)
    P =  torch.tensor(state_data['prices'].transpose(), dtype=torch.float32)
    sigall = torch.load('ex_ante_sigma.pt')

    if asset_type == 'all':
        return S, P, sigall
    else:
        if asset_type == 'commodity':
            ac = 'Commodities'
        elif asset_type == 'fixed_income':
            ac = 'Fixed Income'
        elif asset_type == 'currency':
            ac = 'Currencies'
        elif asset_type == 'index':
            ac = 'Equities'
        else:
            return None

        asset_classes = pd.read_csv('asset_classes.csv')
        asset_idx = asset_classes[asset_classes['asset_class']==ac].index

        S = S[asset_idx] # n_assets * n_days * 60 * 7
        P = P[asset_idx] # n_assets * n_days
        sigall = sigall[asset_idx] # n_assets * n_days

        return S, P, sigall

if __name__ == '__main__':
    data = pd.read_csv('cleaned_data.csv')
    # initial time index
    t0 = len(data)-252*5*3
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
