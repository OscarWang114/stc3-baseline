# -*- coding: utf-8 -*-


class Annotation(object):

    @staticmethod
    def list():
        return ["O", "T", "X"]

    @staticmethod
    def calc_distribution(annotations):
        count_O = 0
        count_T = 0
        count_X = 0

        for annotation in annotations:
            if annotation['breakdown'] == 'O':
                count_O += 1
            elif annotation['breakdown'] == 'T':
                count_T += 1
            elif annotation['breakdown'] == 'X':
                count_X += 1

        prob_O = count_O * 1.0 / (count_O + count_T + count_X)
        prob_T = count_T * 1.0 / (count_O + count_T + count_X)
        prob_X = count_X * 1.0 / (count_O + count_T + count_X)

        return [prob_O, prob_T, prob_X]

    @staticmethod
    def majority_label(prob_O, prob_T, prob_X):

        if prob_O >= prob_T and prob_O >= prob_X:
            return "O"
        elif prob_T >= prob_O and prob_T >= prob_X:
            return "T"
        elif prob_X >= prob_T and prob_X >= prob_O:
            return "X"
        else:
            return "O"
