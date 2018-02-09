import math

import keras
#import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
from keras.models import Model
from .plot.plotter import Plot

if K.backend() == 'theano':
    if K.image_data_format() == 'channels_last':
        K.set_image_data_format('channels_first')
else:
    if K.image_data_format() == 'channels_first':
        K.set_image_data_format('channels_last')


class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''

    def __init__(self, iterations, epochs=1):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''
        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}
        self.best = {}

    def calc_lr(self):
        return K.get_value(self.model.optimizer.lr)

    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        # assign new learning rate to variable.
        K.set_value(self.model.optimizer.lr, self.calc_lr())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self, size=(8, 6)):
        plt = Plot()
        plt.fig = size
        plt.grid = (1, 1)
        plt.subfigure(0, 'Learning Rate Finder', 'Iterations', 'Learning Rate')
        #plt.plot(0, name='', x=self.history['iterations'], y=self.history['lr'], color='b')
        plt.plot(0, self.history['iterations'], self.history['lr'], 'b')
        plt.show()

    def plot(self, size=(8, 6), log_scale=True, n_skip=10):
        plt = Plot()
        plt.fig = size
        plt.grid = (1, 1)
        plt.subfigure(0, 'Learning Rate Finder In Loss', 'Learning Rate (log scale)', 'Loss')
        plt.plot(0, self.history['lr'][n_skip:-1], self.history['loss'][n_skip:-1], 'b')
        label = 'iter: %d \n loss: %.3f \n lr: %.4f' % (self.best['iteration'], self.best['loss'], self.best['lr'])
        print(self.best)
        plt.axs[0].annotate(label, 
                            xy=(self.best['lr'], self.best['loss']), 
                            xytext=(self.best['lr'] * 0.7, self.best['loss'] * 5),
                            arrowprops=dict(facecolor='yellow', shrink=0.05, headwidth=7, width=2.5),
                            horizontalalignment='left',)
        if log_scale:
            plt.axs[0].set_xscale("log")
            #plt.axs[0].set_yscale("log")
        plt.show()


class LR_Find(LR_Updater):
    '''This callback is utilized to determine the optimal lr to be used
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai/conv_learner.py
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    it loosely implements methods described in the paper https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self, iterations, epochs=1, min_lr=1e-05, max_lr=10, jump=6, linear=False):
        '''
        iterations = dataset size / batch size
        epochs should always be 1
        min_lr is the starting learning rate
        max_lr is the upper bound of the learning rate
        jump is the x-fold loss increase that will cause training to stop (defaults to 6)
        '''
        self.linear = linear
        self.min_lr = min_lr
        self.max_lr = max_lr
        ratio = max_lr / min_lr
        self.lr_mult = (ratio / iterations) if linear else ratio ** (1 / iterations)
        #self.lr_mult = (max_lr / min_lr) ** (1 / iterations)
        self.jump = jump
        super().__init__(iterations, epochs=epochs)

    def calc_lr(self):
        mult = self.lr_mult * self.trn_iterations if self.linear else self.lr_mult ** self.trn_iterations
        #lr = self.min_lr * (self.lr_mult ** self.trn_iterations)
        lr = self.min_lr * mult
        return lr

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs=logs)
        try:  # multiple lr's
            K.get_variable_shape(self.model.optimizer.lr)[0]
            self.min_lr = np.full(K.get_variable_shape(self.model.optimizer.lr), self.min_lr)
        except IndexError:
            pass
        K.set_value(self.model.optimizer.lr, self.min_lr)
        self.best = {'iteration':0, 'loss':1e9, 'lr':self.min_lr}
        print(self.best)
        
        self.model.save_weights('tmp.hd5')  # save weights

    def on_train_end(self, logs=None):
        self.model.load_weights('tmp.hd5')  # load_weights

    def on_batch_end(self, batch, logs=None):
        # check if we have made an x-fold jump in loss and training should stop
        try:
            loss = self.history['loss'][-1]
            if math.isnan(loss) or loss > self.best['loss'] * self.jump:
                print('\nbest loss: %.6f' % self.best['loss'])
                print('best lr: %.6f' % self.best['lr'])
                print('best iteration: %d' % self.best['iteration'])
                self.model.stop_training = True
            if loss < self.best['loss']:
                self.best['loss'] = loss
                self.best['iteration'] = self.trn_iterations
                self.best['lr'] = K.get_value(self.model.optimizer.lr)
        except KeyError:
            pass
        super().on_batch_end(batch, logs=logs)


class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement `Cyclical Learning Rates`
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    it loosely implements methods described in the paper https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self, iterations, cycle_len=1, cycle_mult=1, epochs=1):
        '''
        iterations = dataset size / batch size
        epochs #todo do i need this or can it accessed through self.model
        cycle_len = num of times learning rate anneals from its max to its min in an epoch
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations, epochs=epochs)

    def calc_lr(self):
        self.cycle_iterations += 1
        cos_out = np.cos(np.pi * (self.cycle_iterations) / self.epoch_iterations) + 1

        if self.cycle_iterations == self.epoch_iterations:
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        return self.max_lr / 2 * cos_out

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={})  # changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)


class SGD2(optimizers.Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        split1_layer: first middle layer (uses 2nd learning rate)
        split2_layer: first top layer (uses final/3rd learning rate)
        lr: float >= 0. List of Learning rates. [Early layers, Middle layers, Final Layers]
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, split1_layer, split2_layer, lr=[0.0001, .001, .01], momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(optimizers.Optimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.split1 = split1_layer.weights[0].name
            self.split2 = split2_layer.weights[0].name
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    @keras.optimizers.interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        #         print(type(self.lr))
        #         [print(type(item),item.name) for item in params]
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        grp = 0  # set layer grp to 1
        for p, g, m in zip(params, grads, moments):
            if self.split1 == p.name:
                grp = 1
            if self.split2 == p.name:
                grp = 2
            #             print("lr_grp",grp)
            v = self.momentum * m - lr[grp] * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr[grp] * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def finetune(model, num_classes):
    '''removes the last layer of a nn and adds a fully-connected layer for predicting num_classes
    '''
    # model.layers.pop()
    for layer in model.layers: layer.trainable = False
    last = model.layers[-1].output
    x = keras.layers.Flatten()(last)
    preds = keras.layers.Dense(num_classes, activation='softmax', name='fc_start')(x)
    return Model(model.input, preds)


def finetune2(model, num_classes, pool_layer_name, dropout=[.25, .25], dense=1024):
    '''removes the last layers of a nn and adds a fully-connected layers 
    for predicting num_classes
    
    # Arguments
        model: model to finetune
        pool_layer: pooling layer after the final conv layers
            *note this will be replaced by a AvgMaxPoolConcatenation
    '''
    pool_layer = [layer for layer in model.layers if layer.name == pool_layer_name][0]
    model = Model(model.input, pool_layer.input)
    model.layers.pop()
    for layer in model.layers: layer.trainable = False
    last = model.output
    a = keras.layers.MaxPooling2D(pool_size=(7, 7), name='maxpool')(last)
    b = keras.layers.AveragePooling2D(pool_size=(7, 7), name='avgpool')(last)
    x = keras.layers.concatenate([a, b], axis=1)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization(epsilon=1e-05, name='fc_start')(x)
    x = keras.layers.Dropout(dropout[0])(x)
    x = keras.layers.Dense(dense, activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-05)(x)
    x = keras.layers.Dropout(dropout[1])(x)
    preds = keras.layers.Dense(num_classes, activation='softmax')(x)
    return Model(model.input, preds)
