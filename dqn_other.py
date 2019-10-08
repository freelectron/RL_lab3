import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        out = self.l2(x)

        return out

class algo_DQN():
    def __init__(self):
        self.batch_size = 64
        self.discount_factor = 0.8
        self.learn_rate = 1e-3
        # Let's instantiate and test if it works
        num_hidden = 128
        self.epsilon = 0.1
        torch.manual_seed(1234)
        self.model = QNetwork(num_hidden)
        self.optimizer = optim.Adam(self.model.parameters(), self.learn_rate)

    def select_action(self, model, state, epsilon):
        with torch.no_grad():
            actions = model(torch.FloatTensor(state))
            argmax = torch.max(actions, 0)[1]

            action = np.random.choice(2,1,
                        p = [epsilon / 2 + ((1-epsilon) if i == argmax else 0)
                                 for i in range(2)
                            ])
        return action[0]

    def predict(self, state):
        action = self.select_action(self.model, state, self.epsilon).item()

        return action

    def compute_q_val(self, model, state, action):
        actions = model(torch.FloatTensor(state))
        return actions[range(len(state)), action]


    def compute_target(self, model, reward, next_state, done, discount_factor):
        Qvals = model(next_state)
        target = reward + discount_factor * Qvals.max(1)[0] * (1 - done.float())

        return target


    def _train(self, transitions, model, optimizer, batch_size, discount_factor):
        """
        Main train function.
        """
        state, action, reward, next_state, done = transitions

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.uint8)

        q_val = self.compute_q_val(model, state, action)

        # Semi-gradient (no grads wrt target)
        with torch.no_grad():
            target = self.compute_target(model, reward, next_state, done, discount_factor)

        loss = F.smooth_l1_loss(q_val, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, state, actions, rewards, next_state, done):
        """
        Wrapper around _train() to incorporate into Nil's structure.
        """
        loss = self._train((state, actions, rewards, next_state, done), self.model, self.optimizer, self.batch_size, self.discount_factor)

        return loss