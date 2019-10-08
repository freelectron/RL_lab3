import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.modules import MSELoss
from torch.optim import SGD


class QNetwork(nn.Module):

    def __init__(self, num_input, num_output, num_hidden=64):
        super().__init__()
        layers = [nn.Linear(num_input, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_output)]
        self.network = nn.Sequential(*layers)
        self.d_type = self.network.state_dict()['0.weight'].type()

    def forward(self, x):
        return self.network(x)


class DQN(object):
    def __init__(self, input_size, output_size, optimizer=SGD, loss=MSELoss, lr=0.01, gamma=0.99, epsilon_delta=0.0001, epsilon_min=0.05):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_delta = epsilon_delta
        self.epsilon = 1.0
        self.q_network = QNetwork(input_size, output_size, num_hidden=32)
        self.target_q_network = QNetwork(input_size, output_size, num_hidden=32)
        self.update_target_network()
        self.optimizer = optimizer(self.q_network.parameters(), lr=lr)
        self.loss_function = loss()

    def check_obs(self, obs):
        # Check that obs is a tensor
        if type(obs) == torch.tensor:
            pass
        elif type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        else:
            raise ValueError
        # Check tensor is 2D
        assert len(obs.shape) < 3
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, obs.shape[-1])
        # Check feature dimension is correct
        assert obs.shape[-1] == self.input_size
        # Check data type is correct
        if obs.type() != self.q_network.d_type:
            obs = obs.type(self.q_network.d_type)
        return obs

    def predict(self, obs, eval=False):
        obs = self.check_obs(obs)

        q_values = self.q_network(obs)
        if eval:
            return torch.argmax(q_values).item()

        if random.random() < self.epsilon:
            action = random.randint(0, self.output_size-1)
        else:
            action = torch.argmax(q_values).item()
        return action

    def train(self, obses_t, actions, rewards, obses_tp1, dones):
        # Convert to torch.tensor
        obses_t = self.check_obs(obses_t)
        actions = torch.from_numpy(actions).to(torch.long)
        rewards = torch.from_numpy(rewards).to(torch.float32)
        obses_tp1 = self.check_obs(obses_tp1)
        dones = torch.from_numpy(dones).to(torch.float32)
        batch_range = torch.range(0, actions.shape[0]-1, dtype=torch.long)
        # Calculate loss
        self.optimizer.zero_grad()
        predictions = self.q_network(obses_t)[batch_range, actions]
        targets = rewards + self.gamma * dones * self.target_q_network(obses_tp1)[batch_range, actions]
        loss = self.loss_function(predictions, targets)
        # Backprop
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.epsilon_min)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
