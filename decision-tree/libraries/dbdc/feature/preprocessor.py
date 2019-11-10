# -*- coding: utf-8 -*-


import os
import logging
import copy


class Preprocessor():

    KEY_COLUMN = "KEY"

    def __init__(self, columns, extractors, filler):
        self.columns = columns
        self.extrs = extractors
        self.filler = filler

    def apply(self, dialogue):
        d = copy.deepcopy(dialogue)
        d.add_features(self.extrs)
        X = d.to_df()
        eval_X = X[X["EVAL_INDEX"] >= 0].reset_index(drop=True)
        return self.filler.apply(eval_X[self.columns]), eval_X[[self.KEY_COLUMN]]
