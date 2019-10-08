import torch
from torch.optim import SGD, Adam
from torch.nn.modules import MSELoss
import numpy as np
from exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from dqn import DQN
from dqn_other import algo_DQN
import gym
import matplotlib.pyplot as plt

def main(params):
    env = gym.make("CartPole-v0")

    if params['buffer'] == ReplayBuffer:
        buffer = ReplayBuffer(params['buffer_size'])
    elif params['buffer'] == PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(params['buffer_size'], params['PER_alpha'])
    else:
        raise ValueError('Buffer type not found.')

    if params['algorithm'] == DQN:
        algorithm = DQN(env.observation_space.shape[0],
                        env.action_space.n,
                        optimizer=params['optimizer'],
                        loss=params['loss_function'],
                        lr=params['lr'],
                        gamma=params['gamma'],
                        epsilon_delta=params['epsilon_delta'],
                        epsilon_min=params['epsilon_min'])
    elif params['algorithm'] == algo_DQN:
        algorithm = algo_DQN()
    else:
        raise ValueError('Algorithm type not found.')
    losses = []
    returns = []
    episodes_length = []
    for i in range(params['episodes']):
        print(i, '/', params['episodes'], end='\r')
        obs_t = env.reset()
        t = 0
        episode_loss = []
        episode_rewards = []
        while True:
            # env.render()
            action = algorithm.predict(obs_t)
            t += 1
            obs_tp1, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            buffer.add(obs_t, action, reward, obs_tp1, done)
            if len(buffer) >= params['batch_size']:
                batch = buffer.sample(params['batch_size'])
                loss = algorithm.train(*batch)
                episode_loss.append(loss)
                if not isinstance(algorithm, algo_DQN):
                    # this func is not implemented for other_DWN
                    algorithm.update_epsilon()
            if done:
                episodes_length.append(t)
                # env.render()
                print('Episode finished in', t, 'steps')
                print('Cumm reward:', np.sum(episode_rewards), 'Loss:', np.mean(episode_loss), 'Epsilon:', algorithm.epsilon)
                break
            obs_t = obs_tp1
        if not isinstance(algorithm, algo_DQN):
            # this func is not implemented for other_DWN
            algorithm.update_target_network()
        losses.append(np.mean(episode_loss))
        returns.append(np.sum(episode_rewards))
    env.close()

    ## ====== Evaluation ========
    # And see the results
    def smooth(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    plt.plot(smooth(episodes_length, 10))
    plt.title('Episode durations per episode')
    plt.show()


if __name__ == '__main__':
    parameters = {'buffer': ReplayBuffer,
                  'buffer_size': 1500,
                  'PER_alpha': 0.6,
                  'PER_beta': 0.4,
                  'algorithm': algo_DQN,
                  'batch_size': 64,
                  'optimizer': SGD,
                  'loss_function': MSELoss,
                  'lr': 0.01,
                  'gamma': 0.99,
                  'epsilon_delta': 0.0001,
                  'epsilon_min': 0.05,
                  'target_network_interval': 500,
                  'environment': 'MountainCarContinuous-v0',
                  'episodes': 700}
    main(parameters)
