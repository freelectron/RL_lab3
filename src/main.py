import torch
from torch.optim import Adam, SGD
from torch.nn.modules import MSELoss
import numpy as np
from experience_replay.exp_replay import ReplayBuffer, PrioritizedReplayBuffer, HindsightReplayBuffer, PrioritizedHindsightReplayBuffer
from algorithms.dqn import DQN
from algorithms.dqn_other import algo_DQN
import gym
import matplotlib.pyplot as plt
from environments.acrobot_custom import CustomAcrobotEnv
from environments.acrobot_simple import SimpleAcrobotEnv
from environments.gridworld_2 import GridworldEnv


def update(algorithm, buffer, params, train_steps):
    batch = buffer.sample(params['batch_size'])
    if type(buffer) == ReplayBuffer or type(buffer) == HindsightReplayBuffer:
        obses_t, a, r, obses_tp1, dones = batch
        loss = algorithm.train(obses_t, a, r, obses_tp1, dones)
    elif type(buffer) == PrioritizedReplayBuffer or type(buffer) == PrioritizedHindsightReplayBuffer:
        obses_t, a, r, obses_tp1, dones, importance_weights, idxs = batch
        loss, losses = algorithm.per_train(obses_t, a, r, obses_tp1, dones, importance_weights)
        buffer.update_priorities(idxs, losses.numpy() + 1e-8)
    else:
        raise ValueError('?????')

    if isinstance(algorithm, algo_DQN):
        return loss
    # this func is not implemented for other_DWN
    algorithm.update_epsilon()
    if train_steps % params['target_network_interval'] == 0:
        algorithm.update_target_network()
    return loss


def add_transitions_to_buffer(transitions, buffer, completion_reward=0.0, special_goal=False):
    if type(buffer) == ReplayBuffer or type(buffer) == PrioritizedReplayBuffer:
        if special_goal:
            for (f_t, g, a, r, f_tp1, _, done) in transitions:
                obs_t = np.hstack((f_t, g))
                obs_tp1 = np.hstack((f_tp1, g))
                buffer.add(obs_t, a, r, obs_tp1, done)
        else:
            for (f_t, g, a, r, f_tp1, done) in transitions:
                obs_t = np.hstack((f_t, g))
                obs_tp1 = np.hstack((f_tp1, g))
                buffer.add(obs_t, a, r, obs_tp1, done)
    if type(buffer) == HindsightReplayBuffer or type(buffer) == PrioritizedHindsightReplayBuffer:
        if special_goal:
            g_prime = transitions[-1][5]
        else:
            g_prime = transitions[-1][4]
        # Replace goal of every transition
        for i, (f_t, _, a, r, f_tp1, _, done) in enumerate(transitions):
            if i == len(transitions) - 1:
                r = completion_reward  # Last transition has its reward replaced
            buffer.add(f_t, g_prime, a, r, f_tp1, done)


def test(algorithm, env, n_tests=5):
    episodes_length = []

    is_goal = True
    print('Testing...                            ')
    for i in range(n_tests):
        # print(i, '/', n_tests, end='\r')
        if isinstance(env, GridworldEnv):
            obs_t, goal = env.perform_reset()
        elif is_goal:
            obs_t, goal = env.reset()
        else:
            obs_t = env.reset()
            goal = np.zeros_like(obs_t)

        t = 0
        while True:
            # env.render()
            action = algorithm.predict(np.hstack((obs_t, goal)), eval=True)
            t += 1
            if isinstance(env, GridworldEnv):
                obs_tp1, reward, done, _ = env.perform_step(action)
            elif is_goal:
                obs_tp1, reward, done, _, goal = env.step(action)
            else:
                obs_tp1, reward, done, _ = env.step(action)

            # termination condition
            if done:
                episodes_length.append(t)
                # print('Episode finished in', t, 'steps')
                break

            obs_t = obs_tp1

    return np.mean(episodes_length)


def main(params):
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    # declare environment
    is_goal = True
    if params['environment'] == 'acrobot_custom':
        env = CustomAcrobotEnv(stochastic=False, max_steps=200)
        s, goal = env.reset()
    elif params['environment'] == 'acrobot_simple':
        env = SimpleAcrobotEnv(stochastic=False, max_steps=200)
        s, goal = env.reset()
    elif params['environment'] == 'windy_grid_world':
        env = GridworldEnv()
        s, goal = env.perform_reset()
    else:
        env = gym.make(params['environment'])
        s = env.reset()
        goal = s
        is_goal = False

    state_shape = s.shape[0] + goal.shape[0]

    # select type of experience replay using the parameters
    if params['buffer'] == ReplayBuffer:
        buffer = ReplayBuffer(params['buffer_size'])
        loss_function = params['loss_function']()
    elif params['buffer'] == PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(params['buffer_size'], params['PER_alpha'], params['PER_beta'])
        loss_function = params['loss_function'](reduction='none')
    elif params['buffer'] == HindsightReplayBuffer:
        buffer = HindsightReplayBuffer(params['buffer_size'])
        loss_function = params['loss_function']()
    elif params['buffer'] == PrioritizedHindsightReplayBuffer:
        buffer = PrioritizedHindsightReplayBuffer(params['buffer_size'], params['PER_alpha'], params['PER_beta'])
        loss_function = params['loss_function'](reduction='none')
    else:
        raise ValueError('Buffer type not found.')

    # select learning algorithm using the parameters
    if params['algorithm'] == DQN:
        algorithm = DQN(state_shape,
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
    episodes_length_test = []

    print('Starting to train:', type(algorithm), type(buffer))
    test_lengths = test(algorithm, env)
    episodes_length_test.append(test_lengths)
    print(train_steps)

    while train_steps < params['train_steps']:
        if isinstance(env, GridworldEnv):
            obs_t, goal = env.perform_reset()
        elif is_goal:
            obs_t, goal = env.reset()
        else:
            obs_t = env.reset()
            goal = np.zeros_like(obs_t)

        t = 0
        episode_loss = []
        episode_rewards = []
        episode_transitions = []
        while train_steps < params['train_steps']:
            # env.render()
            action = algorithm.predict(np.hstack((obs_t, goal)))
            t += 1
            if isinstance(env, GridworldEnv):
                obs_tp1, reward, done, _ = env.perform_step(action)
            elif is_goal:
                obs_tp1, reward, done, _, gr = env.step(action)
                transition = (obs_t, goal, action, reward, obs_tp1, gr, done)
            else:
                obs_tp1, reward, done, _ = env.step(action)
                transition = (obs_t, goal, action, reward, obs_tp1, done)
            episode_transitions.append(transition)
            episode_rewards.append(reward)
            if len(buffer) >= params['batch_size']:
                loss = update(algorithm, buffer, params, train_steps)
                train_steps += 1
                episode_loss.append(loss)
                if train_steps % params['test_every'] == 0:
                    print(train_steps)
                    test_lengths = test(algorithm, env)
                    episodes_length_test.append(test_lengths)
            # termination condition
            if done:
                episodes_length.append(t)
                # print('Episode finished in', t, 'steps')
                # loss_print = np.mean(episode_loss) if episode_loss else 'NaN'
                # print('Cum. reward:', np.sum(episode_rewards), 'Loss:', loss_print, 'Epsilon:', algorithm.epsilon)
                break

            obs_t = obs_tp1

        add_transitions_to_buffer(episode_transitions, buffer, completion_reward=0.0, special_goal=is_goal)
        losses.append(np.mean(episode_loss))
        returns.append(np.sum(episode_rewards))

    env.close()
    return episodes_length_test, returns, losses


def plot_results(er, per, her, pher, params, episode_avg=10):
    er_returns = np.array([np.array(r) for (r, _, _) in er])
    # er_returns = np.array([np.mean(np.array(r).reshape((-1, episode_avg)), axis=1) for (r, _, _) in er])
    # er_returns = np.concatenate((np.zeros((er_returns.shape[0], 1)), er_returns), axis=1)

    x = np.arange(0, er_returns.shape[1]*params['test_every'], params['test_every'])
    y = np.mean(er_returns, axis=0)
    color = 'blue'
    plt.plot(x, y, color=color, label='Experience Replay')
    y_std = np.std(er_returns, axis=0)
    plt.fill_between(x, y+y_std, y-y_std, color=color, alpha=0.4)
    # plt.errorbar(x, y, yerr=np.std(er_returns, axis=0), capsize=5, ecolor=color, color=color, label='ER')

    if per is not None:
        per_returns = np.array([np.array(r) for (r, _, _) in per])
        # per_returns = np.concatenate((np.zeros((er_returns.shape[0], 1)), per_returns), axis=1)
        y = np.mean(per_returns, axis=0)
        color = 'orange'
        plt.plot(x, y, color=color, label='Prioritized Experience Replay')
        # plt.errorbar(x, y, yerr=np.std(per_returns, axis=0), capsize=5,  ecolor=color, color=color, label='PER')
        y_std = np.std(er_returns, axis=0)
        plt.fill_between(x, y+y_std, y-y_std, color=color, alpha=0.4)

    if her is not None:
        her_returns = np.array([np.array(r) for (r, _, _) in her])
        # her_returns = np.concatenate((np.zeros((er_returns.shape[0], 1)), her_returns), axis=1)
        y = np.mean(her_returns, axis=0)
        color = 'green'
        plt.plot(x, y, color=color, label='Hindsight Experience Replay')
        # plt.errorbar(x, y, yerr=np.std(her_returns, axis=0), capsize=5, ecolor=color, color=color, label='HER')
        y_std = np.std(er_returns, axis=0)
        plt.fill_between(x, y+y_std, y-y_std, color=color, alpha=0.4)

    if pher is not None:
        pher_returns = np.array([np.array(r) for (r, _, _) in pher])
        # pher_returns = np.concatenate((np.zeros((er_returns.shape[0], 1)), pher_returns), axis=1)
        y = np.mean(pher_returns, axis=0)
        color = 'red'
        plt.plot(x, y, color=color, label='Prioritized Hindsight Experience Replay')
        # plt.errorbar(x, y, yerr=np.std(pher_returns, axis=0), capsize=5, ecolor=color, color=color, label='PHER')
        y_std = np.std(er_returns, axis=0)
        plt.fill_between(x, y+y_std, y-y_std, color=color, alpha=0.4)

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Episode return')
    plt.show()
    quit()
    er_losses = np.array([np.array(l) for (_, l) in er])
    per_losses = np.array([np.array(l) for (_, l) in per])
    her_losses = np.array([np.array(l) for (_, l) in her])
    pher_losses = np.array([np.array(l) for (_, l) in pher])


if __name__ == '__main__':
    n = 5
    parameters = {'buffer': ReplayBuffer,
                  'buffer_size': 1500,
                  'PER_alpha': 0.6,
                  'PER_beta': 0.4,
                  'algorithm': DQN,
                  'batch_size': 64,
                  'hidden_size': (64,),
                  'optimizer': SGD,
                  'loss_function': MSELoss,
                  'lr': 1e-3,
                  'gamma': 0.8,
                  'epsilon_delta': 1e-2,
                  'epsilon_min': 0.10,
                  'target_network_interval': 100,
                  'environment': 'acrobot_simple',
                  'episodes': 120}

    er_results = [main(parameters) for _ in range(n)]

    parameters['buffer'] = PrioritizedReplayBuffer
    per_results = [main(parameters) for _ in range(n)]

    parameters['buffer'] = HindsightReplayBuffer
    her_results = [main(parameters) for _ in range(n)]

    parameters['buffer'] = PrioritizedHindsightReplayBuffer
    pher_results = [main(parameters) for _ in range(n)]
    plot_results(er_results, per_results, her_results, pher_results, parameters)
