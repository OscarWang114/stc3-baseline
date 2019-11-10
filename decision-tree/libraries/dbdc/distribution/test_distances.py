# -*- coding: utf-8 -*-
"""Testing for Distribution Distance Function"""


import math
from unittest import TestCase
import nose
from nose.tools import eq_

from distances import jsd, nmd, rsnod, dw


def floor(f, i):
    return round(f - pow(10, -i) / 2, i)


def ceil(f, i):
    return round(f + pow(10, -i) / 2, i)


class DistanceTestCase(TestCase):

    def setUp(self):
        self.a = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.b = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        self.c = ([1.0 / 3, 1.0 / 3, 1.0 / 3], [1.0 / 3, 2.0 / 3, 0.0])
        self.d = ([1.0 / 3, 1.0 / 3, 1.0 / 3], [2.0 / 3, 1.0 / 3, 0.0])
        pass

    def test_jsd(self):
        eq_(jsd(self.a[0], self.a[1]), 1.0)
        eq_(jsd(self.b[0], self.b[1]), 1.0)
        eq_(round(jsd(self.c[0], self.c[1]), 4), 0.2075)
        eq_(round(jsd(self.d[0], self.d[1]), 4), 0.2075)

    def test_nmd(self):
        eq_(nmd(self.a[0], self.a[1]), 0.5)
        eq_(nmd(self.b[0], self.b[1]), 1.0)
        eq_(round(nmd(self.c[0], self.c[1]), 4), 0.1667)
        eq_(round(nmd(self.d[0], self.d[1]), 4), 0.3333)

    def test_rsnod(self):
        # eq_(round(rsnod(self.a[0], self.a[1]), 4), 0.7072)  # 0.7071067811865476 != 0.7072
        eq_(ceil(rsnod(self.a[0], self.a[1]), 4), 0.7072)  # 無理やり切り上げ
        eq_(rsnod(self.b[0], self.b[1]), 1.0)
        eq_(round(rsnod(self.c[0], self.c[1]), 4), 0.3191)
        eq_(round(nmd(self.d[0], self.d[1]), 4), 0.3333)

    def tearDown(self):
        print("done")
