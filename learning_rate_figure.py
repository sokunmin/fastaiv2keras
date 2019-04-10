from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from numpy import arange, sin, pi
import numpy as np

fig = figure(1)
ax = fig.add_subplot(111)
# ---------------------------
gamma = 0.99994
step_size = 20.0
min_lr = 0.001
max_lr = 0.006
iterations = np.arange(0, 200, 1.0)

# https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules_advanced.html
# https://www.jeremyjordan.me/nn-learning-rate/
# https://github.com/bckenstler/CLR
# https://nbviewer.jupyter.org/github/coxy1989/clr/blob/master/notebooks/schedulers.ipynb
def triangular():
    cycle = np.floor(1 + iterations / (2 * step_size))
    x = np.abs(iterations / step_size - 2 * cycle + 1)

    print(x)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x))
    ax.plot(iterations, lr)
    plt.show()


def exp_triangular():
    cycle = np.floor(1 + iterations / (2 * step_size))
    x = np.abs(iterations / step_size - 2 * cycle + 1)
    # x = np.abs(np.sin(iterations / step_size) - 2 * cycle + 1)
    # x = np.abs(np.sin(iterations / step_size) - 2 * cycle + 1)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x)) * (gamma ** iterations)
    ax.plot(iterations, lr)
    plt.show()


# exp_triangular()
# triangular()

def cosine_smooth():
    # https://stackoverflow.com/questions/51926684/plotting-sum-of-two-sinusoids-in-python
    max_steps = 200
    step_size = max_steps / 6.0
    n_cycles = max_steps / step_size
    cycle = np.floor(1 + iterations / (2 * step_size))
    print(type(cycle))
    # 100 / 12 = 0.0833, 1-0.0833=0.91667
    # 12 = 6 x 2 (segments)
    xt = np.linspace(-0.0833, 0.91667, num=max_steps)
    yt = 1 - (np.sin(np.pi * xt * n_cycles) / 2 + 0.5)

    # > [no exp decay]
    # lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - yt)) * (gamma ** iterations)

    # > [half decay]
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - yt)) / 2 ** (cycle - 1)
    ax.plot(iterations, lr)
    plt.show()


cosine_smooth()
