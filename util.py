import torch
import numpy as np
import pandas as pd

# Import main packages and define auxilliary functions

# convert 1D dataframe to tensor
def df2tensor(df):
    return torch.tensor(np.array(df))
