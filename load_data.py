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

    # remove columns with too many NaNs starting from 2004
    data = data[data.index >= '2004']
    nan_lens = data.agg(max_nan_len)
    cols = nan_lens[nan_lens <= 20].index
    data = data[cols]

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
