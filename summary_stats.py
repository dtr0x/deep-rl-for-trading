import pickle
import pandas as pd
import numpy as np
import re

def start_date(data, col):
    return data[col].dropna().index[0]

def daily_returns(data, col):
    prices = np.array(data[col].dropna())
    rets = (prices[1:] - prices[0:-1])/prices[0:-1] * 100
    return rets

if __name__ == '__main__':
    # load dataframe stored in pkl file
    with open('raw.pkl', 'rb') as f:
        data = pickle.load(f)

    symbols = []
    start_dates = []
    mean_rets = []
    std_rets = []
    min_rets = []
    max_rets = []

    for col in data.columns:
        # get symbol
        if '1' in col:
            s = col.split('1')[0].strip()
            symbols.append(s)
        else:
            # columns with 'LM'
            symbols.append(col[2:4])

        # get start date
        start_dates.append(start_date(data, col))

        # get returns
        rets = daily_returns(data, col)
        # get mean returns
        mean_rets.append(np.mean(rets))
        # get std dev
        std_rets.append(np.std(rets))
        # get min
        min_rets.append(np.min(rets))
        #get max
        max_rets.append(np.max(rets))

    summary = pd.DataFrame(columns=['symbol', 'start_date', 'mean', 'std', 'min', 'max'])

    summary['symbol'] = symbols
    summary['start_date'] = start_dates
    summary['mean'] = mean_rets
    summary['std'] = std_rets
    summary['min'] = min_rets
    summary['max'] = max_rets

    s_old = pd.read_csv('summary_stats.csv').drop(columns=['Asset', 'End Date']).rename(columns={'Start Date': 'start_date', 'Symbol': 'symbol', 'Asset Class': 'asset_class'})

    merged = summary.merge(right=s_old, how='inner', on='symbol')
