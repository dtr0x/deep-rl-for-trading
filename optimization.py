import torch
from torch.nn.functional import smooth_l1_loss as huber, mse_loss

''' optimization for DQN training '''

def optimize_model(optimizer, memory, policy_net, target_net, batch_size, \
                    discount_factor, loss_fn=huber):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)

    states = torch.stack([t.state for t in transitions])
    actions = torch.stack([t.action for t in transitions])
    rewards = torch.stack([t.reward for t in transitions])
    next_states = torch.stack([t.next_state for t in transitions])

    # get expected values from target net
    next_state_values, _ = target_net(next_states).max(dim=1)
    # use .detach() to avoid gradient backpropogation on target net
    expected_state_action_values = rewards + discount_factor*next_state_values.detach()

    # get values of currrent actions from policy net
    action_values = policy_net(states)
    action_idx = actions + 1 # action indexes
    row_idx = torch.arange(action_values.size(0))
    action_values = action_values[row_idx, action_idx]

    loss = loss_fn(action_values, expected_state_action_values)

    optimizer.zero_grad() # set gradients to 0 to avoid accumulation
    loss.backward() # backpropograte to compute gradient wrt parameters

    # clip gradients to avoid exploding gradient (even though LSTM is used)
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp(-1, 1)

    optimizer.step() # perform gradient descent step

    return loss
