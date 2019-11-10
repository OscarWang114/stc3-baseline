# -*- coding: utf-8 -*-

import os
import logging
import math


def kld(p, q):
    # Kullback–Leibler divergence
    k = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            k += p[i] * (math.log(p[i] / q[i], 2))

    return k


def jsd(p, q):
    m = []
    for i in range(len(p)):
        m.append((p[i] + q[i]) / 2.0)

    return (kld(p, m) + kld(q, m)) / 2.0


def mse(p, q):
    total = 0.0

    for i in range(len(p)):
        total += pow(p[i] - q[i], 2)

    return total / len(p)


def cp(p, i):
    return sum(p[:i])


def md(p, q):
    total = 0.0

    for i in range(len(p)):
        total += abs(cp(p, i) - cp(q, i))

    return total


def nmd(p, q):
    return md(p, q) / (len(p) - 1)


def dw(p, i, q):
    # Only used here
    total = 0.0

    for j in range(len(q)):
        total += abs(i - j) * pow(p[j] - q[j], 2)

    return total


def od(p, q):
    # Only used here
    total = 0.0
    B_count = 0

    for i in range(len(q)):
        if q[i] > 0:
            B_count += 1
            total += dw(p, i, q)

    return total / B_count


def sod(p, q):
    # Only used here
    return (od(p, q) + od(q, p)) / 2


def rsnod(p, q):
    return math.sqrt(sod(p, q) / (len(p) - 1))
