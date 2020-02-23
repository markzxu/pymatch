#!/usr/bin/env python
# coding: utf-8

from layers import *
from torch import randperm, zeros
import torch
from collections import defaultdict


class Optimizer(object):
    def __init__(self, model, data, lr=0.001, bs=64, metrics=[]):
        """
        metric should be a tuple of (name, function)
        """
        self.model = model
        ((X_train, Y_train), (X_valid, Y_valid)) = data
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.lr = lr
        self.bs = bs
        self.metrics = metrics
    
    def minibatches(self, shuffle=True):
        perm = randperm(self.X_train.size(1))
        self.X_train = self.X_train[:, perm]
        self.Y_train = self.Y_train[:, perm]
        for i in range(0, self.X_train.size(1), self.bs):
            X = self.X_train[:, i:i + self.bs]
            Yhat = self.Y_train[:, i:i + self.bs]
            yield (X, Yhat)
            
    def report_metrics(self, metric_data):
        for k, v in metric_data.items():
            print(f"{k}: {v:.4f}")
    
    def fit(self, epoch_threshold=5, max_epochs=20):
        best_loss = np.inf
        best_model = None
        k = 0
        n = 0
        while k < epoch_threshold and n < max_epochs:
            train_metrics = self.fit_one_epoch()
            valid_metrics = self.valid_loss()
            loss = valid_metrics['valid_loss']
            if loss < best_loss:
                k = 0
                best_loss = loss
                best_model = self.model.clone()
            else:
                k += 1
            n += 1
            self.report_metrics(train_metrics)
            self.report_metrics(valid_metrics)
        return best_model, best_loss
    
    def valid_loss(self):
        metric_data = defaultdict(float)
        Y = self.model(self.X_valid)
        l = self.model.loss(self.Y_valid).item()
        metric_data["valid_loss"] += l 
        for name, metric in self.metrics:
            metric_data["valid_" + name] += metric(Y, self.Y_valid) / self.X_valid.size(-1)
        return metric_data
        
    def fit_one_epoch(self):
        metric_data = defaultdict(float)
        n = 0
        running_loss = 0
        for i, (X, Yhat) in enumerate(self.minibatches()):
            n += X.size(-1)
            Y = self.model(X)
            l = self.model.loss(Yhat)
            metric_data["train_loss"] += l * X.size(-1)
            self.model.backwards()        
            for name, metric in self.metrics:
                metric_data["train_" + name] += metric(Y, Yhat)
            self.step()
            running_loss += l
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (0, i + 1, running_loss / 2000))
                running_loss = 0.0
        for m in metric_data:
            metric_data[m] /= n
        return metric_data


class SGD(Optimizer):
    def step(self):
        for p in self.model.params():
            p -= self.lr * p.grad


class SGD_with_momentum(Optimizer):
    def __init__(self, *args, alpha=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.Vs = None
            
    def step(self):
        if self.Vs is None:
            self.Vs = [-self.lr * p.grad for p in self.model.params()]
        else:
            for p, v in zip(self.model.params(), self.Vs):
                v *= self.alpha
                v -= self.lr * p.grad
        for p, v in zip(self.model.params(), self.Vs):
            p += v


class AdaGrad(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 1e-7# for numerical stabalization
        self.r = [zeros(p.size()) for p in self.model.params()]
    
    def step(self):
        for p, r in zip(self.model.params(), self.r):
            r += p.grad.pow(2)
            p -= (self.lr / (self.delta + r.sqrt())) * p.grad


class ADAM(Optimizer):
    def __init__(self, *args, p1=0.9, p2=0.999, **kwargs):
        super().__init__(*args, **kwargs)
        self.p1 = p1
        self.p2 = p2
        self.delta = 1e-8
        self.t = 0
        self.S = [zeros(p.size()) for p in self.model.params()]
        self.R = [zeros(p.size()) for p in self.model.params()]
    
    def step(self):
        self.t += 1
        for p, s, r in zip(self.model.params(), self.S, self.R):
            s *= self.p1
            s += (1 - self.p1) * p.grad
            
            r *= self.p2
            r += (1 - self.p2) * p.grad.pow(2)
                        
            shat = s /(1 - self.p1 ** self.t)
            rhat = r /(1 - self.p2 ** self.t)
            
            p -= self.lr / np.sqrt(self.t) * shat / (rhat.sqrt() + self.delta)

