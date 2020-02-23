#!/usr/bin/env python
# coding: utf-8

from torch import tensor


def accuracy(x, y):
    return (x.max(0)[1] == y.max(0)[1]).sum().float()

