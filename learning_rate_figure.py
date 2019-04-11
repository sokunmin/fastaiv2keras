from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
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

def get_cycle_range(max_iter, n_cycles, c_mul, paddding=0):
    # 100 / 12 = 0.0833, 1-0.0833=0.91667
    # 12 = 6 x 2 (segments)

    begin = - (100 / (n_cycles * c_mul)) / 100
    end = 1 + begin + paddding
    # return np.linspace(begin, end, num=max_iter)
    return np.linspace(0, 1.0, num=max_iter)


def cosine_smooth2(max_iter=200, n_cycles=2):
    # https://stackoverflow.com/questions/51926684/plotting-sum-of-two-sinusoids-in-python
    # > additional vars for exp_decay
    step_size = max_iter / (2 * n_cycles)
    cycle = np.floor(1 + iterations / (2 * step_size))

    x1 = get_cycle_range(max_iter, n_cycles, 4)
    y1 = np.sin(np.pi * x1 * (n_cycles * 2))

    n_cycles *= 2
    x2 = get_cycle_range(max_iter, n_cycles, 4)
    y2 = np.sin(np.pi * x2 * (n_cycles * 2))

    yt = y2 + y1
    yt = 1 - (yt / 2 + 0.5)

    # > [no exp decay]
    # lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - yt)) * (gamma ** iterations)

    # > [half decay]
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, yt) # / 2 ** (cycle - 1)
    ax.plot(iterations, lr)
    plt.show()


# cosine_smooth()
# cosine_smooth2()

def sinusoid():
    t = np.arange(0.0, 2.0, 0.01)

    y1 = np.sin(4 * np.pi * t)
    y2 = np.sin(2 * np.pi * t)
    ax.plot(t, y1 + y2)
    plt.show()

sinusoid()