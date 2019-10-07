import torch
from torch.optim import SGD, Adam
import numpy as np
from exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from dqn import DQN
import gym


def main(params):
    env = gym.make('MountainCar-v0')

    if params['buffer'] == ReplayBuffer:
        buffer = params['buffer'](params['buffer_size'])
    elif params['buffer'] == PrioritizedReplayBuffer:
        buffer = params['buffer'](params['buffer_size'], params['PER_alpha'])
    else:
        raise ValueError('Buffer type not found.')

    if params['algorithm'] == DQN:
        print(env.action_space)
        algorithm = params['algorithm'](env.observation_space[0],
                                        env.action_space.n,
                                        optimizer=params['optimizer'],
                                        lr=params['lr'],
                                        gamma=params['gamma'],
                                        epsilon_delta=params['epsilon_delta'],
                                        epsilon_min=params['epsilon_min'])
    else:
        raise ValueError('Algorithm type not found.')

    for i in range(params['episodes']):
        print(i, '/', params['episodes'], end='\r')
        obs_t = env.reset()
        for t in range(300):
            env.render()
            action = algorithm.predict(obs_t)
            obs_tp1, reward, done = env.step(action)
            buffer.add(obs_t, action, reward, obs_tp1, done)
            batch = buffer.sample(64)
            algorithm.train(*batch)
            if done:
                env.render()
                print('Episode finished in', t, 'steps')
                break
            obs_t = obs_tp1
    env.close()


if __name__ == '__main__':
    parameters = {'buffer': ReplayBuffer,
                  'buffer_size': 1000,
                  'PER_alpha': 0.6,
                  'PER_beta': 0.4,
                  'algorithm': DQN,
                  'batch_size': 64,
                  'optimizer': SGD,
                  'lr': 0.001,
                  'gamma': 0.99,
                  'epsilon_delta': 0.001,
                  'epsilon_min': 0.05,
                  'target_network_interval': 500,
                  'environment': 'MountainCarContinuous-v0',
                  'episodes': 1000}
    main(parameters)
