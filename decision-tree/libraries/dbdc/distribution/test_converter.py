# -*- coding: utf-8 -*-
"""Testing for Distribution Converter"""


from unittest import TestCase
import nose
from nose.tools import eq_


import pandas as pd


from converter import DistributionConverter, StatDistributionConverter


class DistributionConverterTestCase(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "TURN_INDEX": [0, 1, 2, 3, 4],
            "PROB-O": [1.0, 0.0, 0.0, 1.0 / 3, 1.0],
            "PROB-T": [0.0, 1.0, 0.0, 1.0 / 3, 1.0],
            "PROB-X": [0.0, 0.0, 1.0, 1.0 / 3, 1.0],
        })
        self.con = DistributionConverter()
        pass

    def test_convert(self):
        df2 = self.con.convert(self.df)
        eq_(len(df2), len(self.df))
        eq_(df2.columns.tolist(), ["PROB-O", "PROB-T", "PROB-X"])

    def test_restore(self):
        df2 = self.con.restore(self.df)
        eq_(len(df2), len(self.df))
        eq_(df2.columns.tolist(), ["PROB-O", "PROB-T", "PROB-X"])
        eq_(df2.values.sum(), 5.0)
        eq_(df2.iloc[4]["PROB-O"], 1.0 / 3)

    def tearDown(self):
        print("done")


class StatDistributionConverterTestCase(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "TURN_INDEX": [0, 1, 2, 3, 4],
            "PROB-O": [1.0, 0.0, 0.0, 1.0 / 3, 2.0 / 3],
            "PROB-T": [0.0, 1.0, 0.0, 1.0 / 3, 1.0 / 3],
            "PROB-X": [0.0, 0.0, 1.0, 1.0 / 3, 0.0],
        })
        self.con = StatDistributionConverter()
        pass

    def test_convert(self):
        df2 = self.con.convert(self.df)
        eq_(len(df2), len(self.df))
        eq_(df2.columns.tolist(), ["AVE", "VAR2"])
        eq_(df2.iloc[0]["AVE"], -1.0)

    # TODO
    # def test_restore(self):
    #     pass

    def tearDown(self):
        print("done")
