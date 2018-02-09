# Copyright 2017 by Chun-Ming Su

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plot(object):
    def __init__(self):
        self._fig = None
        self._gs = None
        self._projection = None
        self._axs = {}

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, figsize):
        self._fig = plt.figure(figsize=figsize)

    @property
    def grid(self):
        return self._gs

    @grid.setter
    def grid(self, spec):
        self._gs = gridspec.GridSpec(spec[0], spec[1])

    @property
    def axs(self):
        return self._axs
        
    def subfigure(self, index, title, xlabel, ylabel, zlabel=None, projection='2d'):
        self._projection = projection
        ax = self._fig.add_subplot(self._gs[index], projection=projection) \
            if projection == '3d' else self._fig.add_subplot(self._gs[index])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
        self._axs[index] = ax

    # def plot(self, index, name, x, y, color, alpha=1.0):
    #     self._axs[index].plot(x, y, label=name, c=color, linewidth=1, alpha=alpha)

    def plot(self, index, *args):
        self._axs[index].plot(*args)
        
    def hist(self, ax, data, bins, range, rwidth, color):
        return ax.hist(data, bins=bins, range=range, rwidth=rwidth, color=color)[0]

    def scatter(self, ax, data, color, marker='o', s=10, linewidths=1, alpha=0.5, zorder=10):
        if self._projection == '3d':
            return ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                              marker=marker,
                              s=s,
                              color=color,
                              linewidths=linewidths,
                              alpha=alpha,
                              zorder=zorder)
        else:
            return ax.scatter(data[:, 0], data[:, 1],
                              marker=marker,
                              s=s,
                              color=color,
                              linewidths=linewidths,
                              alpha=alpha,
                              zorder=zorder)

    def legend(self, handles, labels, loc, ncol, bbox_to_anchor):
        plt.figlegend(handles=handles, labels=labels, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    def show(self, tight_pad=None, save_name=None):
        if tight_pad:
            plt.tight_layout(tight_pad)
        if save_name:
            plt.savefig(save_name)
        plt.show()
