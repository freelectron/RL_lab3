from torch.optim import Adam
from torch.nn.modules import MSELoss
import numpy as np
from experience_replay.exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from algorithms.dqn import DQN
from algorithms.dqn_other import algo_DQN
import gym
import matplotlib.pyplot as plt


def update(algorithm, buffer, params, train_steps):
    batch = buffer.sample(params['batch_size'])
    if type(buffer) == ReplayBuffer:
        obses_t, a, r, obses_tp1, dones = batch
        loss = algorithm.train(obses_t, a, r, obses_tp1, dones)
    elif type(buffer) == PrioritizedReplayBuffer:
        obses_t, a, r, obses_tp1, dones, importance_weights, idxs = batch
        loss, losses = algorithm.per_train(obses_t, a, r, obses_tp1, dones, importance_weights)
        buffer.update_priorities(idxs, losses.numpy() + 1e-8)
    else:
        raise ValueError('?????')
    if not isinstance(algorithm, algo_DQN):
        # this func is not implemented for other_DWN
        algorithm.update_epsilon()
        if train_steps % params['target_network_interval'] == 0:
            algorithm.update_target_network()
    return loss


def main(params):
    env = gym.make('CartPole-v0')

    if params['buffer'] == ReplayBuffer:
        buffer = ReplayBuffer(params['buffer_size'])
        loss_function = params['loss_function']()
    elif params['buffer'] == PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(params['buffer_size'], params['PER_alpha'], params['PER_beta'])
        loss_function = params['loss_function'](reduction='none')
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
    elif params['algorithm'] == algo_DQN:
        algorithm = algo_DQN()
    else:
        raise ValueError('Algorithm type not found.')
    losses = []
    returns = []
    train_steps = 0
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
                train_steps += 1
                loss = update(algorithm, buffer, params, train_steps)
                episode_loss.append(loss)
            if done:
                episodes_length.append(t)
                # env.render()
                print('Episode finished in', t, 'steps')
                print('Cumm reward:', np.sum(episode_rewards), 'Loss:', np.mean(episode_loss), 'Epsilon:', algorithm.epsilon)
                break
            obs_t = obs_tp1
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
    parameters = {'buffer': PrioritizedReplayBuffer,
                  'buffer_size': 1500,
                  'PER_alpha': 0.6,
                  'PER_beta': 0.4,
                  'algorithm': DQN,
                  'batch_size': 64,
                  'hidden_size': (64,),
                  'optimizer': Adam,
                  'loss_function': MSELoss,
                  'lr': 1e-3,
                  'gamma': 0.8,
                  'epsilon_delta': 1e-4,
                  'epsilon_min': 0.10,
                  'target_network_interval': 50,
                  'environment': 'MountainCarContinuous-v0',
                  'episodes': 400}
    main(parameters)
