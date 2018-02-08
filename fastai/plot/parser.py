########################################################################
#
# Published under the MIT License. See the file LICENSE for details.
# Copyright 2017 by Chun-Ming Su
#
########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os
import numpy as np
import scipy.ndimage.filters
import scipy.interpolate

# https://stackoverflow.com/questions/41135078/matplotlib-make-smooth-graph-line
parse_txt = lambda file: np.loadtxt(file, unpack=True, delimiter=',')


class Parser(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def parse(self, dir, files, slice_size=None):
        pass

    @abstractmethod
    def compute_mean(self):
        pass

    @abstractmethod
    def smooth(self, sigma):
        pass

    def _is_iterable(self, val):
        return isinstance(val, (tuple, list))


# -------------------------------------------
class AccuracyParser(Parser):

    def __init__(self, name):
        super(AccuracyParser, self).__init__(name)
        self.steps = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.error = []

    def parse(self, dir, files, slice_size=None):
        for file in files:
            f = os.path.join(dir, file)
            steps, accuracy, precision, recall, f1_score, error = parse_txt(f)
            self.steps = steps[:slice_size]
            self.accuracy.append(accuracy[:slice_size])
            self.precision.append(precision[:slice_size])
            self.recall.append(recall[:slice_size])
            self.f1_score.append(f1_score[:slice_size])
            self.error.append(error[:slice_size])

    def data(self, x_type, y_type):
        x = self._type(x_type)
        y = self._type(y_type)

        return self.name, x, y

    def _type(self, dtype):
        if dtype == "error":
            data = self.error
        elif dtype == "accuracy":
            data = self.accuracy
        elif dtype == "f1_score":
            data = self.f1_score
        elif dtype == "steps":
            data = self.steps
        else:
            raise ValueError("Incorrect argument")
        return data

    def compute_mean(self):
        self.accuracy = np.mean(self.accuracy, axis=0)
        self.precision = np.mean(self.precision, axis=0)
        self.recall = np.mean(self.recall, axis=0)
        self.f1_score = np.mean(self.f1_score, axis=0)
        self.error = np.mean(self.error, axis=0)

    def smooth(self, sigma):
        self.accuracy = scipy.ndimage.filters.gaussian_filter1d(self.accuracy, axis=0, sigma=sigma)
        self.precision = scipy.ndimage.filters.gaussian_filter1d(self.precision, axis=0, sigma=sigma)
        self.recall = scipy.ndimage.filters.gaussian_filter1d(self.recall, axis=0, sigma=sigma)
        self.f1_score = scipy.ndimage.filters.gaussian_filter1d(self.f1_score, axis=0, sigma=sigma)
        self.error = scipy.ndimage.filters.gaussian_filter1d(self.error, axis=0, sigma=sigma)


# -------------------------------------------
class HistogramParser(Parser):

    def __init__(self, name):
        super(HistogramParser, self).__init__(name)
        self.defects = []
        self.non_defects = []

    def parse(self, dir, files, slice_size=None):
        for file in files:
            f = os.path.join(dir, file)
            defects, non_defects = parse_txt(f)
            self.defects.append(defects[:slice_size])
            self.non_defects.append(non_defects[slice_size:])

    def compute_mean(self):
        self.defects = np.mean(self.defects, axis=0)
        self.non_defects = np.mean(self.non_defects, axis=0)

    def smooth(self, sigma):
        pass