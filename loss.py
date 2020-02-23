#!/usr/bin/env python
# coding: utf-8

from torch import tensor, zeros
from layers import Softmax


class Loss(object):
    def __call__(self, y, yhat):
        self.input = y
        self.target = yhat
        return self.forward(y, yhat)
    
    def forward(self, y, yhat):
        raise Exception("not implemented")
    
    def backwards(self):
        raise Exception("not implemented")


class SquaredErrorLoss(Loss):
    def forward(self, y, yhat):
        return (y - yhat).pow(2).mean()
    
    def backwards(self):
        self.input.grad = (2 * (self.input - self.target))/self.target.size(1)


class NegativeLogLikelihoodLoss(Loss):
    def forward(self, y, yhat):
        return -((yhat * y.log()).sum(0)).mean()
    
    def backwards(self):
        self.input.grad = -(self.target - self.input)/((1 - self.input) * self.input)/self.target.size(1)


class CrossEntropyLoss(Loss):
    # composition of Softmax and NegativeLogLikelihoodLoss
    # prefered for reasons of numerical stability and
    # don't have to compute as many gradients
    
    def __init__(self):
        self.f = Softmax()
        
    def forward(self, y, yhat):
        return ((-y * yhat).sum(0) + y.exp().sum(0).log()).mean()
    
    def backwards(self):
        self.input.grad = (self.f(self.input) - self.target) / self.target.size(-1)

