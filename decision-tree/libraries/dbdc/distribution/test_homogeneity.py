# -*- coding: utf-8 -*-
"""Testing for Distribution Distance Function"""


from unittest import TestCase
import nose
from nose.tools import eq_

import numpy as np

from homogeneity import Homogeneity


def check_sum(distributions):
    for dist in distributions:
        sum = np.sum(dist)
        # sum = round(np.sum(dist), 4)
        if sum != 1.0:
            raise Exception("1.0 != sum(%s)" % dist)


class HomogeneityTestCase(TestCase):

    def setUp(self):
        self.distribution_set_one = [[1.0 / 3, 1.0 / 3, 1.0 / 3],
                                     [1.0 / 3, 1.0 / 3, 1.0 / 3]]
        self.distribution_set_two = [[1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0]]
        self.distribution_set_three = [[0.0, 1.0, 0.0],
                                       [0.0, 1.0, 0.0]]

        check_sum(self.distribution_set_one)
        check_sum(self.distribution_set_two)
        check_sum(self.distribution_set_three)
        pass

    def test_call(self):
        nbin = 3
        homo = Homogeneity(nbin=nbin)

        eq_(homo(self.distribution_set_one), 0.0)
        eq_(homo(self.distribution_set_two), 0.0)
        eq_(homo(self.distribution_set_three), 0.0)

    def tearDown(self):
        print("done")
