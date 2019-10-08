import torch
from torch.optim import SGD, Adam
from torch.nn.modules import MSELoss
import numpy as np
from exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from dqn import DQN
import gym


def update(algorithm, buffer, params, train_steps):
    batch = buffer.sample(params['batch_size'])
    if type(buffer) == ReplayBuffer:
        obses_t, a, r, obses_tp1, dones = batch
        loss = algorithm.train(obses_t, a, r, obses_tp1, dones)
    elif type(buffer) == PrioritizedReplayBuffer:
        obses_t, a, r, obses_tp1, dones, importance_weights, idxs = batch
        batch_loss = algorithm.per_train(obses_t, a, r, obses_tp1, dones, importance_weights)
        buffer.update_priorities(idxs, batch_loss.numpy() + 1e-8)
        loss = batch_loss.mean().item()
    else:
        raise ValueError('?????')
    algorithm.update_epsilon()
    if train_steps % params['target_network_interval'] == 0:
        algorithm.update_target_network()
    return loss


def main(params):
    env = gym.make('CartPole-v0')

    if params['buffer'] == ReplayBuffer:
        buffer = ReplayBuffer(params['buffer_size'])
        loss_function = MSELoss()
    elif params['buffer'] == PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(params['buffer_size'], params['PER_alpha'])
        loss_function = MSELoss(reduction='none')
    else:
        raise ValueError('Buffer type not found.')

    if params['algorithm'] == DQN:
        algorithm = DQN(env.observation_space.shape[0],
                        env.action_space.n,
                        loss_function=loss_function,
                        optimizer=params['optimizer'],
                        lr=params['lr'],
                        gamma=params['gamma'],
                        epsilon_delta=params['epsilon_delta'],
                        epsilon_min=params['epsilon_min'])
    else:
        raise ValueError('Algorithm type not found.')
    losses = []
    returns = []
    train_steps = 0
    for i in range(params['episodes']):
        print(i, '/', params['episodes'], end='\r')
        obs_t = env.reset()
        t = 0
        episode_loss = []
        episode_rewards = []
        while True:
            env.render()
            action = algorithm.predict(obs_t)
            t += 1
            obs_tp1, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            buffer.add(obs_t, action, reward, obs_tp1, done)
            if len(buffer) >= params['batch_size']:
                train_steps += 1
                loss = update(algorithm, buffer, params, train_steps)
                episode_loss.append(loss)
            if done:
                env.render()
                print('Episode finished in', t, 'steps')
                print('Cumm reward:', np.sum(episode_rewards), 'Loss:', np.mean(episode_loss), 'Epsilon:', algorithm.epsilon)
                break
            obs_t = obs_tp1

        losses.append(np.mean(episode_loss))
        returns.append(np.sum(episode_rewards))
    env.close()


if __name__ == '__main__':
    parameters = {'buffer': ReplayBuffer,
                  'buffer_size': 1000,
                  'PER_alpha': 0.6,
                  'PER_beta': 0.4,
                  'algorithm': DQN,
                  'batch_size': 64,
                  'optimizer': SGD,
                  'loss_function': MSELoss,
                  'lr': 0.01,
                  'gamma': 0.99,
                  'epsilon_delta': 0.0001,
                  'epsilon_min': 0.05,
                  'target_network_interval': 200,
                  'environment': 'MountainCarContinuous-v0',
                  'episodes': 1000}
    main(parameters)
