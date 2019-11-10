# -*- coding: utf-8 -*-

import os
import logging
import numpy as np


try:
    from .distances import jsd
except Exception as e:
    print(" Exception: %s" % e)
    from distances import jsd


class Comprehensiveness(object):

    @staticmethod
    def meanvar(distributions):
        vars = np.var(distributions, axis=0)
        mean_var = np.mean(vars)
        return mean_var

    @staticmethod
    def disvar(distributions):
        mean = np.mean(distributions, axis=0)
        jsds = []
        for dist in distributions:
            jsds.append(jsd(dist, mean))
        return np.mean(jsds)
