import torch
from torch.nn.functional import smooth_l1_loss as huber

''' optimization for DQN training '''

def optimize_model(optimizer, memory, policy_net, target_net, batch_size, \
                    discount_factor):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)

    states = torch.stack([t.state for t in transitions])
    actions = torch.cat([t.action for t in transitions])
    next_states = torch.stack([t.next_state for t in transitions])
    rewards = torch.cat([t.reward for t in transitions])

    # get expected values from target net
    next_state_values, _ = target_net(next_states).max(dim=1)
    # use .detach() to avoid gradient backpropogation on target net
    expected_state_action_values = rewards + discount_factor*next_state_values.detach()

    # get values of currrent actions from policy net
    action_values = policy_net(states)
    action_idx = actions.long() + 1 # action indexes
    row_idx = torch.arange(action_values.size(0))
    action_values = action_values[row_idx, action_idx]

    loss = huber(action_values, expected_state_action_values)

    optimizer.zero_grad() # set gradients to 0 to avoid accumulation
    loss.backward() # backpropograte to compute gradient wrt parameters

    optimizer.step() # perform gradient descent step

    return loss
