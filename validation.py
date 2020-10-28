import torch
from policy_net import PolicyNet
from save_states_all import load_states
from reinforcement import reward
import

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet().to(device)
    params = torch.load('models/dqn_epoch_20.pt')
    model.load_state_dict(params)
    model.eval()

    S, P, sigall = [x.to(device) for x in load_states('commodity')]

    val_idx = range(2500,4000)

    actions_prev = torch.zeros(len(S), device=device)

    bp = torch.tensor(2e-3, device=device)
    tgt_vol = torch.tensor(0.1, device=device)

    returns = []
    for t in val_idx:
        states = S[:, t]
        next_states = S[:, t+1]
        prices = P[:, t]
        prices_next = P[:, t+1]
        sigmas = sigall[:, t]

        actions = model.get_actions(states)
        rewards = reward(prices, prices_next, sigmas, actions, actions_prev, tgt_vol, bp)
        returns.append(rewards.mean().item())

    crets = np.cumsum(returns)
