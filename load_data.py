import pickle
from util import *

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

    # forward filling all NA values
    data = data.reset_index(drop=True)
    n = data.shape[0]
    k = data.shape[1]-1
    colnames = data.columns[1:]
    for tt in range(0,k):
        data_vec = data[colnames[tt]]
        data_vec = data_vec.reset_index(drop=True)
        data_vec_na = data_vec.isna()
        for i in range(1,n):
            if data_vec_na[i] == True:
                data[colnames[tt]][i] = data[colnames[tt]][i-1]

    # saving data to csv
    data.to_csv('cleaned_data.csv', index=False)
