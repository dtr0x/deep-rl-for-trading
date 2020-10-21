import torch
import numpy as np
import pandas as pd
from state_space import state_space

if __name__ == '__main__':
    # test commodoties data
    # data = pd.read_csv('cleaned_data.csv').drop(columns='date')
    # cols = [c for c in data.columns if 'Comdty' in c]
    # data = data[cols]
    # states = state_space(data)

    n_assets = 10
    n_days = 1000
    seq_length = 60
    features = 7

    states = torch.rand(n_assets, n_days, seq_length, features)
