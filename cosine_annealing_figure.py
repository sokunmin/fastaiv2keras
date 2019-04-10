from fastai.plot.plotter import Plot
import numpy as np

plt = Plot()
plt.fig = (10, 5)
plt.grid = (1, 1)
plt.subfigure(0, 'Cosine Annealing', 'Iteration', 'Learning Rate')


def iteration(begin, end):
    x = np.arange(begin, end, 1)
    y = 1 + np.cos(np.pi * (x / end))
    if begin == 0:
        x = np.concatenate(([0], x), axis=0)
        y = np.concatenate(([0.001], y), axis=0)
    return x, y


cycle = 100

x1, y1 = iteration(0, cycle)

x2, y2 = iteration(0, (cycle * 2))
x2 = x2 + cycle

x3, y3 = iteration(0, (cycle * 4))
x3 = x3 + (cycle * 3)

x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

plt.plot(0, x, y, color='r')
plt.show(tight_pad=2.1)