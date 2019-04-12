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

def get_cycle_range2(max_iters, n_cycles, c_mul):
    # 100 / 12 = 0.0833, 1-0.0833=0.91667
    # 12 = 6 x 2 (segments)

    begin = - (100 / (n_cycles * c_mul)) / 100
    end = 1 + begin
    return np.linspace(begin, end, num=max_iters)
    # return np.arange(begin, end, 0.01)


def get_cycle_range(max_iters, n_cycles, c_mul):

    begin = - (max_iters / (n_cycles * c_mul)) / 100
    end = 1 + begin
    return np.linspace(begin, end, num=max_iters)



def cosine_smooth2(max_iters=200, n_cycles=3):
    # https://stackoverflow.com/questions/51926684/plotting-sum-of-two-sinusoids-in-python
    # > additional vars for exp_decay
    step_size = max_iters / (2 * n_cycles)
    cycle = np.floor(1 + iterations / (2 * step_size))

    x1 = get_cycle_range(max_iters, n_cycles, 4)
    yt = np.sin(np.pi * x1 * (n_cycles * 2))

    yt = 1 - (yt / 2 + 0.5)

    # > [no exp decay]
    # lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - yt)) * (gamma ** iterations)

    # > [half decay]
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, yt) / 2 ** (cycle - 1)
    ax.plot(iterations, lr)
    plt.show()


def sinusoid(max_iters=200, n_cycles=2, c_mul=6):
    # x = np.arange(0.0, 2.0, 0.01)
    x = get_cycle_range(max_iters, n_cycles, c_mul)

    print('len(t): ', len(x))
    y1 = np.sin(n_cycles * np.pi * x)
    y2 = np.sin(n_cycles * 2 * np.pi * x)
    y = (y1 + y2)
    max_y, min_y = max(y), min(y)
    y = y / (max_y + np.abs(min_y))
    y = y + 0.5

    lr = min_lr + (max_lr - min_lr) * np.maximum(0, y)# / 2 ** (cycle - 1)

    ax.plot(iterations, lr)
    plt.show()


def sinusoid_with_decay(max_iter=200, n_cycles=3):
    # https://stackoverflow.com/questions/51926684/plotting-sum-of-two-sinusoids-in-python
    step_size = max_iter / (2 * n_cycles)
    cycle = np.floor(1 + iterations / (2 * step_size))

    # 100 / 12 = 0.0833, 1-0.0833=0.91667
    # 12 = 6 x 2 (segments)
    xt = get_cycle_range2(max_iter, n_cycles, 4)
    # xt = np.linspace(-0.0833, 0.91667, num=max_iter)
    yt = 1 - (np.sin(np.pi * xt * (2 * n_cycles)) / 2 + 0.5)

    # > [no exp decay]
    # lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - yt)) * (gamma ** iterations)

    # > [half decay]
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - yt)) / 2 ** (cycle - 1)
    ax.plot(iterations, lr)
    plt.show()


sinusoid_with_decay()
# sinusoid()


