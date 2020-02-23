#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch import tensor, einsum, zeros


class Layer(object):
    def __init__(self):
        pass
    
    def forward(self, x):
        raise Exception("not implemented")
        
    def __call__(self, x):
        self.input = x
        self.out = self.forward(x)
        return self.out
    
    def approx_grad(self, p, epsilon=1e-4):
        """
        approximates the limit definition of the gradient
        w.r.t. p
        
        assumes that self.out.grad is all ones, i.e. that 
        the loss function is a simple sum over outputs
        
        idea from http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
        """
        grad = zeros(p.size())
        for i in range(len(p)):
            for j in range(len(p[i])):
                p[i, j] += epsilon
                right = self.forward(self.input)
                p[i, j] -= 2 * epsilon
                left = self.forward(self.input)
                p[i, j] += epsilon
                grad[i, j] = ((right.sum() - left.sum()) / (2 * epsilon))
        return grad
        
    def backwards(self):
        raise Exception("not implemented")
        
    def params(self):
        return None
    
    def clone(self):
        return self


class Linear(Layer):
    def __init__(self, w, b, init=True):
        self.w = w
        self.b = b
        if init:
            # W_ij ~ U (-sqrt(6/(m+n)), sqrt(6/(m + n) as per Goodfellow
            w_init = np.sqrt(6 / sum(w.size()))
            self.w.uniform_(-w_init, w_init)
            self.b.uniform_(-w_init, w_init) # default value of approx 0.1 as per Goodfellow
    
    def forward(self, x):
        return self.w@x + self.b
    
    def backwards(self):
        self.input.grad = self.w.t()@self.out.grad
        self.w.grad = (self.input.unsqueeze(0) * self.out.grad.unsqueeze(1)).sum(-1)
        self.b.grad = self.out.grad.sum(1).unsqueeze(1)
    
    def params(self):
        return [self.w, self.b]
    
    def clone(self):
        return Linear(self.w.clone(), self.b.clone())


class ReLU(Layer):
    def forward(self, x):
        return x.clamp(0, np.inf)
    
    def backwards(self):
        self.input.grad = (self.input > 0).float() * self.out.grad


class Softmax(Layer):
    def forward(self, x):
        out = x - x.max(0)[0] # numerical stability
        out.exp_()
        out /= out.sum(0)
        return out
    
    def backwards(self):
        I = tensor(np.identity(self.out.size(0))).float()
        I = I.unsqueeze(-1).repeat(1, 1, self.input.size(1))
        k = self.out.T.unsqueeze(-1) * (I - self.out).permute(2, 0, 1)
        self.input.grad = einsum("ijk,ki->ji", [k, self.out.grad])


class Model(Layer):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss_func = loss
    
    def clone(self):
        return Model([l.clone() for l in self.layers], self.loss)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def loss(self, yhat):
        return self.loss_func(self.out, yhat)
    
    def backwards(self):
        self.loss_func.backwards()
        for l in reversed(self.layers):
            l.backwards()
    
    def params(self):
        params = []
        for l in self.layers:
            p = l.params()
            if p is None:
                continue
            params += p
        return params




