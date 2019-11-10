# -*- coding: utf-8 -*-


from collections import defaultdict
import pandas as pd


class DFCreator(dict):

    def __init__(self, names=[]):
        dict.__init__(self)
        for name in names:
            self[name] = defaultdict(list)

    def add(self, name, row_dict):
        for k, v in row_dict.items():
            self[name][k].append(v)

    def to_df(self, name):
        return pd.DataFrame(self[name])
