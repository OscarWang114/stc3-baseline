# -*- coding: utf-8 -*-

import os
import logging
import numpy as np

try:
    from .distances import jsd
except Exception as e:
    print(" Exception: %s" % e)
    from distances import jsd


class Homogeneity(object):

    def __init__(self, nbin=3, dist_func=jsd):
        self.nbin = nbin
        self.f = dist_func
        self.q = np.diag(np.ones(nbin))

    def __call__(self, distributions):
        scores = []
        for dist in distributions:
            scores.append(self.score(dist))
        return np.mean(scores)

    def score(self, dist):
        positive_distance = self.f(dist, self.q[0])
        negative_distance = self.f(dist, self.q[-1])
        return positive_distance - negative_distance
