import pickle
import numpy as np

if __name__ == '__main__':
    with open('df.pkl', 'rb') as f:
        data = pickle.load(f)

    # get commodities data as numpy array
    cols = [c for c in data.columns if 'Comdty' in c]
    comms = data[cols].values.transpose()
