import numpy as np
import matplotlib.pyplot as plt

def plot_per(result_dict, params, fig_name='figure'):
    colors = ['blue', 'orange', 'green', 'red', 'pink']
    for i, name in enumerate(result_dict):

        returns = np.array([np.array(r) for (r, _, _) in result_dict[name]])

        x = np.arange(0, returns.shape[1]*params['test_every'], params['test_every'])
        y = np.mean(returns, axis=0)
        color = colors[i]
        plt.plot(x, y, color=color, label='Experience Replay')
        y_std = np.std(returns, axis=0)
        plt.fill_between(x, y+y_std, y-y_std, color=color, alpha=0.4)

    plt.legend()
    plt.xlabel('Training steps')
    plt.ylabel('Episode length')
    plt.title('Grid world 9x9')
    plt.savefig('results/' + fig_name + '.png')
    plt.clf()