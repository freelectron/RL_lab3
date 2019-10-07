import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import SGD


class QNetwork(nn.Module):

    def __init__(self, num_input, num_output, num_hidden=64):
        super().__init__()
        layers = [nn.Linear(num_input, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_output)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DQN(object):
    def __init__(self, input_size, output_size, optimizer=SGD, lr=0.001, gamma=0.99, epsilon_delta=0.001, epsilon_min=0.05):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_delta = epsilon_delta
        self.epsilon = 1.0
        self.q_network = QNetwork(input_size, output_size, num_hidden=64)
        self.target_q_network = QNetwork(input_size, output_size, num_hidden=64)
        self.update_target_network()
        self.optimizer = optimizer(self.q_network.parameters(), lr=lr)

    def predict(self, obs, eval=False):
        if type(obs) == torch.tensor:
            pass
        elif type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        else:
            raise ValueError
        assert len(obs.shape) < 3
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, obs.shape[-1])
        assert obs.shape[-1] == self.input_size

        q_values = self.q_network(obs)
        if eval:
            return torch.argmax(q_values)

        if random.random() < self.epsilon:
            action = random.randint(self.output_size)
        else:
            action = torch.argmax(q_values)
        return action

    def train(self, obses_t, actions, rewards, obses_tp1, dones):
        predictions = self.q_network(obses_t)[actions]
        targets = rewards + dones * self.gamma * self.target_q_network(obses_tp1)[actions]
        loss = predictions - targets

        self.optimizer.zero_grad()
        loss.backwards()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.epsilon_min)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
