# -*- coding: utf-8 -*-

import math


class DistributionConverter(object):

    COLUMNS = ["NB", "PB", "B"]

    def convert(self, dist):
        return dist

    def restore(self, predicted_targets):
        s = sum(predicted_targets)
        return [p / s for p in predicted_targets]


class StatDistributionConverter(object):

    COLUMNS = ["AVE", "VAR"]

    MAP = [-1.0, 0.0, 1.0]

    def convert(self, dist):
        ave = sum([p * v for p, v in zip(dist, self.MAP)])
        var2 = sum([p * math.pow((v - ave), 2) for p, v in zip(dist, self.MAP)])
        return [ave, var2]

    def restore(self, predicted_targets):

        ave = predicted_targets[0]
        var2 = predicted_targets[1]

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
