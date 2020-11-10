import pickle
import pandas as pd

# return the maximum length period of empty data from price series
def max_nan_len(prices):
    pna = prices.isna()
    max_na_count = 0
    na_count = 0
    for i in pna:
        if i:
            na_count += 1
        else:
            if na_count > max_na_count:
                max_na_count = na_count
                na_count = 0
    return max_na_count

# total percentage of NaNs in series
def nan_rate(prices):
    return prices.isna().sum()/len(prices)

# forward fill nan values in price series (in-place)
def forward_fill(prices):
    idx = prices.index
    i = idx[0]
    for j in idx[1:]:
        if pd.isna(prices[j]):
            prices[j] = prices[i]
        i = j
    return prices

if __name__ == '__main__':
    # load dataframe stored in pkl file
    with open('raw.pkl', 'rb') as f:
        data = pickle.load(f)

    # rename columns as symbols
    symbols = []
    data = data.rename(columns = {'SM1 Index': 'SMI1 Index'})
    for col in data.columns:
        # get symbol
        if '1' in col:
            s = col.split('1')[0].strip()
            symbols.append(s)
        else:
            # columns with 'LM'
            symbols.append(col[2:4])
    data.columns = symbols

    # remove columns with too many total NaNs
    data = data[data.index >= '2004']
    nan_rates = data.agg(nan_rate)
    cols = nan_rates[nan_rates <= 0.05].index
    data = data[cols]
    # remove columns with too many consecutive NaNs
    nan_lens = data.agg(max_nan_len)
    cols = nan_lens[nan_lens <= 20].index.sort_values()
    data = data[cols]

    # save asset types from summary stats file
    summary = pd.read_csv('summary_stats.csv').rename(columns={'Symbol': 'symbol',
    'Asset Class': 'asset_class'})[['symbol', 'asset_class']]
    # remove whitespace
    summary['symbol'] = [s.strip() for s in summary['symbol']]
    summary['asset_class'] = [a.strip() for a in summary['asset_class']]
    summary = summary[summary['symbol'].isin(cols)].sort_values(by='symbol')
    summary.to_csv('asset_classes.csv', index=False)

    # get first date where no columns have NaNs
    init_date = data.dropna().index[0]
    # reset first date
    data = data[data.index >= init_date]

    # forward fill all asset columns
    data.agg(forward_fill)

    # make 'date' index a column
    data = data.reset_index()
    # remove 'ticker' column name
    data.columns.name = None
    # saving data to csv
    data.to_csv('cleaned_data.csv', index=False)
