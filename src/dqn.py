import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.modules import MSELoss
from torch.optim import SGD


class QNetwork(nn.Module):

    def __init__(self, num_input, num_output, num_hidden=(64,)):
        super().__init__()
        layers = []
        layer_n = (num_input, *num_hidden, num_output)
        for i in range(len(layer_n[:-1])):
            layers.append(nn.Linear(layer_n[i], layer_n[i+1]))
            if i < len(layer_n) - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.d_type = self.network.state_dict()['0.weight'].type()

    def forward(self, x):
        return self.network(x)


class DQN(object):
    def __init__(self, input_size, output_size, loss_function, num_hidden=(64,), optimizer=SGD, lr=0.001, gamma=0.99, epsilon_delta=0.0001, epsilon_min=0.05):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_delta = epsilon_delta
        self.epsilon = 1.0
        self.q_network = QNetwork(input_size, output_size, num_hidden=num_hidden)
        self.target_q_network = QNetwork(input_size, output_size, num_hidden=num_hidden)
        self.update_target_network()
        self.optimizer = optimizer(self.q_network.parameters(), lr=lr)
        self.loss_function = loss_function

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
        with torch.no_grad():
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
        obses_t = torch.from_numpy(obses_t).to(torch.float32)
        actions = torch.from_numpy(actions).to(torch.long)
        rewards = torch.from_numpy(rewards).to(torch.float32)
        obses_tp1 = torch.from_numpy(obses_tp1).to(torch.float32)
        dones = torch.from_numpy(dones).to(torch.float32)
        batch_range = torch.arange(0, actions.shape[0], dtype=torch.long)
        # Calculate loss
        predictions = self.q_network(obses_t)[batch_range, actions]
        targets = rewards + self.gamma * (1 - dones) * self.target_q_network(obses_tp1).max(1)[0].detach()
        loss = self.loss_function(predictions, targets)
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def per_train(self, obses_t, actions, rewards, obses_tp1, dones, importance_weights):
        # Convert to torch.tensor
        obses_t = torch.from_numpy(obses_t).to(torch.float32)
        actions = torch.from_numpy(actions).to(torch.long)
        rewards = torch.from_numpy(rewards).to(torch.float32)
        obses_tp1 = torch.from_numpy(obses_tp1).to(torch.float32)
        dones = torch.from_numpy(dones).to(torch.float32)
        importance_weights = torch.from_numpy(importance_weights).to(torch.float32)
        batch_range = torch.arange(0, actions.shape[0], dtype=torch.long)
        # Calculate loss
        self.optimizer.zero_grad()
        predictions = self.q_network(obses_t)[batch_range, actions]
        targets = rewards + self.gamma * (1 - dones) * self.target_q_network(obses_tp1).max(1)[0].detach()
        losses = self.loss_function(predictions, targets)
        loss = (losses * importance_weights).mean()
        # Backprop
        loss.backward()
        self.optimizer.step()
        return loss.item(), torch.abs(predictions - targets).detach()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.epsilon_min)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
