import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from collections import deque
import random 

class MaxLengthBufferList(deque):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def append(self, item):
        deque.append(self, item)
        if len(self) > self.max_length:
            self.popleft()


class DeepQNetwork(nn.Module):

    def __init__(self, num_features, actions):
        super().__init__()
        self.layer1 = nn.Linear(num_features, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, actions)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x


class DeepQLearner:

    def __init__(self, num_features=22, actions=3, alpha=0.2, gamma=0.9, epsilon=0.98, epsilon_decay=0.999, dyna=0, update_interval=1000):    
        self.device = torch.device("cpu")
        self.num_features = num_features
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        self.q_network = DeepQNetwork(num_features, actions).to(self.device)
        self.t_network = copy.deepcopy(self.q_network).to(self.device)
        self.opt = torch.optim.SGD(self.q_network.parameters(), lr=1e-4)
        self.loss_func = torch.nn.MSELoss()
        self.history = MaxLengthBufferList(max_length=10000)
        self.training_iterations = 0  
        self.update_interval = update_interval
        self.losses = []
        self.prev_state = np.zeros((1, 22))
        self.prev_action = 0

    def update_target_network(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            self.epsilon *= self.epsilon_decay
            return np.random.randint(self.actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            actions = self.q_network(state_tensor.unsqueeze(0)) 
            return torch.argmax(actions).item()

    def train(self, s, r):
        self.training_iterations += 1
        #print("iteration: ", self.training_iterations)

        if self.training_iterations % self.update_interval == 0:
            self.update_target_network()

        self.history.append((self.prev_state, self.prev_action, s, r))

        batch_size = 2000
        if len(self.history) < batch_size:
            action = np.random.randint(self.actions)
            self.prev_state = s
            self.prev_action = action
            return action

        action = self.choose_action(s)

        experiences = random.choices(self.history, k=batch_size) # with replacement
        states, actions, next_states, rewards = zip(*experiences)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        q_values = self.q_network(states).detach()
        next_q_values = self.t_network(next_states).detach()
        q_n_values = self.q_network(next_states).detach()

        rewards = torch.tensor(np.tile(rewards.unsqueeze(1), (1, 3))).unsqueeze(1)

        
        # Compute indices of the maximum values along the last dimension
        max_indices = torch.argmax(q_n_values, dim=2).unsqueeze(1)
        # print("max: ", max_indices)

        # Repeat the indices along the second dimension to match the shape of next_q_values
        max_indices_repeated = max_indices.repeat(1, 1, next_q_values.shape[2])

        # Use gather to select elements from next_q_values using max_indices_repeated
        q_t_max = next_q_values.gather(2, max_indices_repeated)
        
        target_q_values = rewards + self.gamma * q_t_max

        q_values.requires_grad = True

        q_values = q_values.gather(2, actions.unsqueeze(1).unsqueeze(1))

        loss = self.loss_func(q_values, target_q_values[:, :, 0].unsqueeze(1))
        self.losses.append(loss.item())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.prev_state = s
        self.prev_action = action

        return action

    def test(self, s, allow_random=False):
        if allow_random and np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(s, dtype=torch.float32).to(self.device)
                q_values = self.q_network(state_tensor.unsqueeze(0))
                return torch.argmax(q_values).item()



