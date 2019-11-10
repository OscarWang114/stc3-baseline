# -*- coding: utf-8 -*-

import os
import logging
import glob
import json
import math
import codecs
from collections import Counter, defaultdict
import numpy as np


class Dialogue(dict):

    def __init__(self, dialogue_json):
        dict.__init__(self)
        self.update(dialogue_json)

    def tokenized_turns(self, tokenizer):
        turns = []
        for turn_i, turn in enumerate(self["turns"]):
            if turn_i != turn["turn-index"]:
                raise Exception("turn_i != turn['turn-index']")
            speaker = turn["speaker"]
            terms = tokenizer.tokenize(turn["utterance"])
            turns.append({"speaker": speaker, "terms": terms})
        return turns

    def dist_list(self, all=True):
        """
        label order: ["O", "T", "X"]
        """
        dists = []
        for turn in self["turns"]:
            breakdowns = [a["breakdown"] for a in turn["annotations"]]
            if all:
                if len(breakdowns) > 0:
                    counter = Counter(breakdowns)
                    s = float(sum(counter.values()))
                    dist = [counter.get(l, 0) / s for l in ["O", "T", "X"]]
                    dists.append(dist)
                else:
                    dists.append([])
            else:
                if turn["speaker"] == "S" and len(breakdowns) > 0:
                    counter = Counter(breakdowns)
                    s = float(sum(counter.values()))
                    dist = [counter.get(l, 0) / s for l in ["O", "T", "X"]]
                    dists.append(dist)
        return dists

    @staticmethod
    def parse_annotations(annotations):
        labels = [a["breakdown"] for a in annotations]
        c = Counter(labels)
        for l in ["O", "T", "X"]:
            c[l] = c.get(l, 0)
        s = sum(c.values())
        ave = (-1.0 * c["X"] + 1.0 * c["O"]) / s
        var = (math.pow((1.0 - ave), 2) * c["O"] + math.pow(ave, 2)
               * c["T"] + math.pow((-1.0 - ave), 2) * c["X"]) / s

        is_bd = False
        labels = c.most_common()
        if labels[0][0] == "X" and labels[0][1] > labels[1][1]:
            is_bd = True

        return c, ave, var, is_bd

    @staticmethod
    def breakdown_dist(annotations, labels=["O", "T", "X"]):
        breakdowns = [a["breakdown"] for a in annotations]
        counter = Counter(breakdowns)
        s = float(sum(counter.values()))
        if s == 0.0:
            return []

        bdist = [counter.get(l, 0) / s for l in labels]
        is_bd = np.argmax(bdist) == len(bdist) - 1

        return is_bd, bdist

    @staticmethod
    def dist_to_stat(dist, map_values=[-1.0, 0.0, 1.0]):
        ave = sum([p * v for p, v in zip(dist, map_values)])
        var2 = sum([p * math.pow((v - ave), 2) for p, v in zip(dist, map_values)])
        return ave, var2

    @staticmethod
    def calc_breakdown_dist(ave, var, n=30):
        """var must be already squared"""
        ave2 = ave ** 2

        prob_O = (ave2 + ave + var) / 2.0
        prob_T = 1 - ave2 - var
        prob_X = (ave2 - ave + var) / 2.0

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
            raise Exception("O: %f, T: %f, X: %f (ave: %f, var: %f)" %
                            (prob_O, prob_T, prob_X, ave, var))

        return [prob_O, prob_T, prob_X]

    @staticmethod
    def get_keyword_features(keywords, terms):
        if not terms:
            # return [None for _ in range(len(keywords))]
            return [0 for _ in range(len(keywords))]
        else:
            return [1 if k in terms else 0 for k in keywords]

    @staticmethod
    def parse_turns(tokenizer, turns, all=False, before=2, after=0):
        for turn in turns:

            # tokenize utterance
            tokens_list = tokenizer.tokenize(turn["utterance"])
            # concat tokens of sentences
            terms = [token["stemmed"] for tokens in tokens_list for token in tokens]

            turn.update({"terms": terms})

        if all:
            for turn in turns:
                yield turn
        else:
            for turn_i, turn in enumerate(turns):
                if turn_i >= before and turn["speaker"] == "S":

                    from_i = turn_i - before
                    to_i = turn_i + after + 1

                    yield turns[from_i:to_i]

    def terms(self, tokenizer, all=False):
        for corpus, dialogue in self:
            lu_terms = None
            ls_terms = None

            for turn in dialogue["turns"]:
                u = tokenizer.tokenize(turn["utterance"])
                terms = [t["stemmed"] for s in u for t in s]

                if all:
                    yield corpus, dialogue, turn, terms
                else:
                    if turn["speaker"] == "S" and turn["turn-index"] == 0:
                        ls_terms = terms
                    elif turn["speaker"] == "S" and turn["turn-index"] > 0:
                        _terms = {
                            "cs": terms,
                            "lu": lu_terms,
                            "ls": ls_terms,
                        }
                        yield corpus, dialogue, turn, _terms

                        ls_terms = terms
                    elif turn["speaker"] == "U":
                        lu_terms = terms

    def to_df(self):
        data = defaultdict(list)

        eval_index = 0
        for turn in self['turns']:
            data["DIALOGUE_ID"].append(self["dialogue-id"])
            data["TURN_INDEX"].append(turn["turn-index"])
            data["UTTERANCE"].apend(turn["utterance"])
            # data["GOLD_LABEL"].append(ans_label)
            # data["PRED_LABEL"].append(pred_label)

            if turn['speaker'] == "U" or turn['annotations'] == []:  # modified Sep 17 2017
                data["EVAL_INDEX"].apend(-1)  # not None
                for p_i, l in enumerate(["O", "X", "T"]):
                    data["P(%s)" % l].append(None)
                data["LABEL"].append(None)
            else:
                data["EVAL_INDEX"].apend(eval_index)
                eval_index += 1

                prob_dist = calc_distribution(turn['annotations'])
                label = majority_label(prob_dist[0], prob_dist[1], prob_dist[2])

                for p_i, l in enumerate(["O", "X", "T"]):
                    data["P(%s)" % l].append(prob_dist[p_i])

                data["LABEL"].append(label)

                # target_label = eval_json['turns'][eval_index]['labels'][0]
                # pred_prob_dist = [float(target_label['prob-O']),
                #                   float(target_label['prob-T']), float(target_label['prob-X'])]
                # pred_label = target_label['breakdown']

        return pd.DataFrame(data)


class Jsons(list):

    FILENAME = "*"

    CLASS = None

    def __init__(self, json_dir):
        list.__init__(self)

        self.dir = json_dir

        self.files = glob.glob(os.path.join(self.dir, self.FILENAME))

        for f in self.files:
            data_fp = codecs.open(f, "r", "utf-8")
            data_json = json.load(data_fp)
            data_fp.close()

            self.append(self.CLASS(data_json) if self.CLASS else data_json)

    def to_df(self):
        return pd.concat([j.to_df() for j in self])


class Dialogues(Jsons):

    FILENAME = "*log.json"
    CLASS = Dialogue


# class Dialogues(list):
#
#     def __init__(self, json_dir, labels=False):
#         list.__init__(self)
#
#         self.dir = json_dir
#         if not labels:
#             self.ext = "*log.json"
#         else:
#             self.ext = ".labels.json"
#
#         self.files = glob.glob(os.path.join(self.dir, self.ext))
#
#         for f in self.files:
#             data_fp = codecs.open(f, "r", "utf-8")
#             data_json = json.load(data_fp)
#             data_fp.close()
#
#             self.append(Dialogue(data_json))
#
#         return self
#
#     def dataframe(self):
#         pass


class Label(dict):

    def __init__(self, data_json):
        dict.__init__(self)
        self.update(data_json)

    def dataframe(self):
        pass


class Labels(list):

    def __init__(self, json_dir):
        list.__init__(self)

        list.__init__(self)

        self.dir = json_dir
        if not labels:
            self.ext = "*log.json"
        else:
            self.ext = ".labels.json"

        self.files = glob.glob(os.path.join(self.dir, self.ext))

        for f in self.files:
            data_fp = codecs.open(f, "r", "utf-8")
            data_json = json.load(data_fp)
            data_fp.close()

            self.append(Dialogue(data_json))

        return self
