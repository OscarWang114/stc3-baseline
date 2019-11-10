# -*- coding: utf-8 -*-

import math


class DistributionConverter(object):

    SOURCE_COLUMNS = ["PROB-O", "PROB-T", "PROB-X"]
    TARGET_COLUMNS = ["PROB-O", "PROB-T", "PROB-X"]

    def apply(self, sdf, convert_func, target_columns):
        tdf = sdf.apply(convert_func, axis=1, result_type="expand")
        tdf.rename(columns={i: col for i, col in enumerate(target_columns)}, inplace=True)
        return tdf

    def convert(self, df):
        return self.apply(df, self._convert, self.TARGET_COLUMNS)

    def _convert(self, row):
        sources = [row[col] for col in self.SOURCE_COLUMNS]
        return self.convert_func(sources)

    def convert_func(self, sources):
        return sources

    def restore(self, df):
        return self.apply(df, self._restore, self.SOURCE_COLUMNS)

    def _restore(self, row):
        targets = [row[col] for col in self.TARGET_COLUMNS]
        return self.restore_func(targets)

    def restore_func(self, targets):
        s = sum(targets)
        return [t / s for t in targets]


class StatDistributionConverter(DistributionConverter):

    SOURCE_COLUMNS = ["PROB-O", "PROB-T", "PROB-X"]
    TARGET_COLUMNS = ["AVE", "VAR2"]

    MAP = [1.0, 0.0, -1.0]

    def convert_func(self, sources):
        ave = sum([p * v for p, v in zip(sources, self.MAP)])
        var2 = sum([p * math.pow((v - ave), 2) for p, v in zip(sources, self.MAP)])
        return [ave, var2]

    def restore_func(self, targets):
        ave = targets[0]
        var2 = targets[1]

        ave2 = ave ** 2

        prob_O = (ave2 + ave + var2) / 2.0
        prob_T = 1 - ave2 - var2
        prob_X = (ave2 - ave + var2) / 2.0

        if prob_O < 0.0:
            prob_O = 0.0
            prob_T = 1.0 + ave
            prob_X = -ave

        elif prob_T < 0.0:
            if ave > 1.0:
                prob_O = 1.0
                prob_T = 0.0
                prob_X = 0.0
            elif ave < -1.0:
                prob_O = 0.0
                prob_T = 0.0
                prob_X = 1.0
            else:
                prob_O = 0.5 + ave / 2
                prob_T = 0.0
                prob_X = 0.5 - ave / 2

        elif prob_X < 0.0:
            prob_O = ave
            prob_T = 1.0 - ave
            prob_X = 0.0

        else:
            s = prob_O + prob_T + prob_X
            prob_O = prob_O / s
            prob_T = prob_T / s
            prob_X = prob_X / s

        if prob_O < 0.0 or 1.0 < prob_O or prob_T < 0.0 or 1.0 < prob_T or prob_X < 0.0 or 1.0 < prob_X:
            raise Exception("O: %f, T: %f, X: %f (ave: %f, var2: %f)" %
                            (prob_O, prob_T, prob_X, ave, var2))

        return [prob_O, prob_T, prob_X]
