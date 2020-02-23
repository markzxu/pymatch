#!/usr/bin/env python
# coding: utf-8

from torch import tensor
import numpy as np


def load_mnist():
    import mnist
    X_train = tensor(mnist.train_images()/255).view(-1, 784).T
    Y_train = mnist.train_labels()
    Y_train = tensor(np.array([np.array(range(10)) == y for y in Y_train])).T.float()

    X_valid = tensor(mnist.test_images()/255).view(-1, 784).T
    Y_valid = mnist.test_labels()
    Y_valid = tensor(np.array([np.array(range(10)) == y for y in Y_valid])).T.float()

    return ((X_train, Y_train), (X_valid, Y_valid))


def load_simple_linear():
    coeff = tensor(np.random.random((1, 2)))
    X_train = tensor(np.random.random((2, 1000)))
    Y_train = coeff@X_train
    X_valid = tensor(np.random.random((2, 500)))
    Y_valid = coeff@X_valid
    
    return ((X_train, Y_train), (X_valid, Y_valid))




