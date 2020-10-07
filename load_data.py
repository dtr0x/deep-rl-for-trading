import pickle
import numpy as np
import pandas as pd
import torch

# load time series data and save as tensors

if __name__ == '__main__':
    # load dataframe stored in pkl file
    with open('raw.pkl', 'rb') as f:
        data = pickle.load(f)

    # make 'date' index a column
    data = data.reset_index()

    # remove 'ticker' column name
    data.columns.name = None

    # get data after 2004
    data = data[data['date'] >= '2004-01-01']

    # drop columns with too many NaNs
    data = data.drop(columns=['WN1 Comdty', 'QC1 Index', 'XB1 Comdty', 'ST1 Index'])

    # get first date where no columns have NaNs
    init_date = data.dropna()['date'].tolist()[0]

    # reset first date
    data = data[data['date'] >= init_date]

    # TODO: add forward fill code
    # ...
    # ...
    # ...

    # uncomment once forward fill complete
    #data.to_csv('cleaned_data.csv', index=False)
