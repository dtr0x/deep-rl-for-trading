## Deep reinforcement learning for trading

reference: https://arxiv.org/pdf/1911.10107.pdf

### Using this codebase

```
requirements:
- Python >= v3.6
- cuda >= v10.0
- pytorch >= v1.3
- numpy
- matplotlib
- multiprocessing
- pandas
- GPU >= 2GB
```

currently implemented algorithms: 
- DQN with target network and experience replay
- policy gradient (PG) (90% complete)
- advantage actor critic (A2C) (90% complete)

TODO:
- double/dueling DQN
- A2C with continuous action space sampling

### Loading the dataset
The initially provided dataset is contained in `raw.pkl`, and the script `load_data.py` is used for preprocessing such as removing assets with too many NaNs, renaming columns and cross-referencing asset classes from `summary_stats.csv`. The data is trimmed to start at 2004 and the resulting dataframe of 55 assets is stored in `cleaned_data.csv`. We also store the asset classes for lookup later in `asset_classes.csv`. Note that these files are already in the repo and it is not required to reload the data.

### State space transformation 
All asset price series contained in `cleaned_data.csv` are preprocessed to form the entire state space used for RL algorithms. Two files are in play to build the state space: `state_space.py` and `save_states_all.py`. The function `get_states_iter` will construct all states for a given price series and initial time `t0`. The main loop in `save_states_all.py` constructs all states from `t0 = len(data)-252*5*3`, which represents approximately 15 years of data. Note that running `save_states_all.py` is resource intensive, and uses the `multiprocessing` module to compute the state series for all assets in parallel. The current state data is already provided in `state_data.npz`.

#### state_space.py 
The following functions are used to compute the individual state elements, each returning the last 60 values from a given time point `t`:
- `norm_prices` : the normalized close price series
- `norm_returns`: normalized returns for 1-month, 2-month, 3-month, 1-year periods (21,42,63,252 days, respectively). The `pandas` function `ewm.std` is used to compute the exponentially weighted moving standard deviation with 60 day time span as described in the paper.
- `macd`: MACD exactly following the implementation details in the paper. **Note:** there is some ambiguity in the paper stating they *average* MACD signals over different time scales, but they provide a sum in equation (12). We chose to use the average in our implementation.
- `rsi`: Relative strength index following its standard implementation. We chose to scale the RSI between 0 and 1 to be on a similar scale with other features.

#### Loading state data
`save_states_all.py` contains the function `load_states(asset_type)`, where asset type can be 'all', 'commodity', 'fixed_income', 'currency', or 'index'. This will load a tensor of all states (dimension: n_assets x n_days x 60 x 7), the corresponding price series (n_assets x n_days), and the corresponding exponentially weighted moving standard deviations (ewmstd) of additive returns for all days (n_assets x n_days). The ewmstd's are preprocessed in this fashion because they are used to compute rewards during training, and we use pandas for this calculation. Since training occurs on GPU, this preprocessing is more efficient that moving data between CPU and GPU memory during training (only PyTorch functions and tensors can operate on GPU, which excludes pandas). We typically load the data in the following way during training:
> `S, P, sigall = [x.to(device) for x in load_states(asset_type)]`

### Reward function
The reward function is implemented following equation (4) in the paper. It has the following signature, contained in `reinforcement.py`:
> `get_reward(prices, prices_next, sigmas, sigmas_prev, actions, actions_prev, tgt_vol, bp)`

### Network architecture
The network for DQN Q-values and action selection is given in `policy_net.py`, following the implementation of the paper but with standard tanh activation in LSTM cells instead of leaky ReLU.

### Training
To train a DQN, we run:
> `python train_dqn.py -asset_type=[all|commodity|fixed_income|currency|index] -min_year=[2006-2019] -max_year=[2007-2020]`

We typically train each asset class over a 5 year period in order to evaluate on the subsequent 5 years, e.g,
> `python train_dqn.py -asset_type=commodity -min_year=2006 -max_year=2010`

The training proceeds over 20 epochs with early stopping. Validation set and training average returns are reported to standard output after each epoch. The best performing model on the validation set is stored in the `models/` folder. 

### Evaluation
To produce cumulative return plots and compare with benchmark algorithms, see `validation.py`, which is run by 
> `python validation.py -asset_type=[all|commodity|fixed_income|currency|index] -min_year=[2006-2019] -max_year=[2007-2020]`

which will produce the plot in the `plots/` folder. Currently, a trained model for 2006-2010 and 2011-2015 must exist for the given asset class in order to make a plot or the script will throw an error. For example, for 2011-2020, we typically run
> `python validation.py -asset_type=commodity -min_year=2011 -max_year=2020`


