# -*- coding: utf-8 -*-

import os
import logging
import numpy as np

from skato.nlp.utils import OfferWeight, SentenceTfidfSimilarity, SentenceWord2vecSimilarity
from dbdc.base.label import BREAKDOWN
from dbdc.base.annotation import Annotation
from dbdc.eval.utils import majority_label
from dbdc.distribution.homogeneity import Homogeneity


from .stat import StatInfo
from .abstract import AbstractExtractor


class TurnLengthExtractor(AbstractExtractor):

    DOMAIN = "TURN"
    NEED_COLUMNS = ["tokens", "utterance"]
    OUTPUT_COLUMNS = ["t_length", "c_length"]

    def extract(self, turn):
        tokens = [token["token"]  # not stemmed tokens
                  for sent in turn["tokens"] for token in sent]
        return [len(tokens), len(turn["utterance"])]


class DistributionExtractor(AbstractExtractor):

    DOMAIN = "TURN"
    NEED_COLUMNS = ["speaker", "annotations"]
    OUTPUT_COLUMNS = ["prob"]

    def extract(self, turn):
        if turn['speaker'] == "U" or turn['annotations'] == []:  # modified Sep 17 2017
            return [[]]
        else:
            return [Annotation.calc_distribution(turn["annotations"])]


class BreakdownLabelExtractor(AbstractExtractor):

    DOMAIN = "TURN"
    NEED_COLUMNS = ["speaker", "annotations", "prob"]
    OUTPUT_COLUMNS = ["breakdown"]

    def extract(self, turn):
        if turn['speaker'] == "U" or turn['annotations'] == []:  # modified Sep 17 2017
            return "N"
        else:
            # return ["O", "T", "X"][np.argmax(turn["prob"])]
            return majority_label(*turn["prob"])


class HomogeneityExtractor(AbstractExtractor):

    def __init__(self, nbin=3):
        self.name = "homogeneity"
        self.homo = Homogeneity(nbin=nbin)

    def extract(self, turn):
        if len(turn["prob"]) == 3:
            return self.homo.score(turn["prob"])
        else:
            return 0.0


class ProbabilityExtractor(AbstractExtractor):

    DOMAIN = "TURN"
    NEED_COLUMNS = ["prob"]
    OUTPUT_COLUMNS = ["prob-%s" % l for l in BREAKDOWN.LABELS]

    def extract(self, turn):
        if len(turn["prob"]) == 3:
            return turn["prob"]
        else:
            return [-1 for _ in range(3)]


# class DistanceFromDialogueMeanExtractor(Extractor):
#
#     def __init__(self, distance_name, func):
#         self.name = "%s_from_dialogue_mean" % distance_name
#         self.func = func
#
#     def fit(self, dialogue):
#         df = dialogue.to_df()
#         try:
#             eval_df = df[df["EVAL_INDEX"] >= 0]
#         except Exception as e:
#             types = [type(df["EVAL_INDEX"].iloc[i]) for i in range(len(df["EVAL_INDEX"]))]
#             import ipdb
#             ipdb.set_trace()
#         mean_p = eval_df[[("prob-%s" % l).upper() for l in BREAKDOWN.LABELS]].mean()
#         self.mean = [mean_p[("prob-%s" % l).upper()] for l in BREAKDOWN.LABELS]
#
#     def extract(self, turn):
#         if len(turn["prob"]) == 3:
#             prob = [turn["prob-%s" % l] for l in BREAKDOWN.LABELS]
#             try:
#                 d = self.func(prob, self.mean)
#             except Exception as e:
#                 import ipdb
#                 ipdb.set_trace()
#             return self.func(prob, self.mean)
#         else:
#             return -1
#
#
# class DistanceFromMeanExtractor(Extractor):
#
#     def __init__(self, distance_name, func):
#         self.name = "%s_from_corpus_mean" % distance_name
#         self.func = func
#
#     def fit(self, dialogues):
#         df = dialogues.to_df()
#         try:
#             eval_df = df[df["EVAL_INDEX"] >= 0]
#         except Exception as e:
#             types = [type(df["EVAL_INDEX"].iloc[i]) for i in range(len(df["EVAL_INDEX"]))]
#             import ipdb
#             ipdb.set_trace()
#         mean_p = eval_df[[("prob-%s" % l).upper() for l in BREAKDOWN.LABELS]].mean()
#         self.mean = [mean_p[("prob-%s" % l).upper()] for l in BREAKDOWN.LABELS]
#
#     def extract(self, turn):
#         if len(turn["prob"]) == 3:
#             prob = [turn["prob-%s" % l] for l in BREAKDOWN.LABELS]
#             try:
#                 d = self.func(prob, self.mean)
#             except Exception as e:
#                 import ipdb
#                 ipdb.set_trace()
#             return self.func(prob, self.mean)
#         else:
#             return -1


def create_of_from_stat(stat, bef_i):
    return OfferWeight(stat["N"], stat["R"], stat["stemmed_df"], stat["stemmed_rdf"][bef_i])


class FeatureExtractor(AbstractExtractor):

    DOMAIN = "DIALOGUE"

    def __init__(self, tokenizer, model_path, max_bef=3, keyword_n=10, **kwargs):
        self.stat = StatInfo(max_bef=max_bef)
        self.keyword_n = keyword_n

        self.COLUMNS = []

        self.STANDALONE_COLUMNS = []
        self.standalone_feature_flag = kwargs.get("standalone_features", False)
        if self.standalone_feature_flag:
            self.STANDALONE_COLUMNS = ["TURN_I", "C_LENGTH", "T_LENGTH"]
            self.COLUMNS.extend(self.STANDALONE_COLUMNS)

        self.KEYWORD_COLUMNS = []
        self.keyword_feature_flag = kwargs.get("keyword_features", False)
        if self.keyword_feature_flag:
            for bef_i in range(self.stat.max_bef):
                for keyword_i in range(keyword_n):
                    self.KEYWORD_COLUMNS.append("KEYWORD_%02d_%d" % (keyword_i, bef_i))
            self.COLUMNS.extend(self.KEYWORD_COLUMNS)

        self.TSIM_COLUMNS = []
        self.tsim_feature_flag = kwargs.get("tsim_features", False)
        if self.tsim_feature_flag:
            for base_i, target_i in self.pair_gen(self.stat.max_bef):
                self.TSIM_COLUMNS.append("TSIM_%d_%d" % (base_i, target_i))
            self.COLUMNS.extend(self.TSIM_COLUMNS)

        self.WSIM_COLUMNS = []
        self.wsim_feature_flag = kwargs.get("wsim_features", False)
        if self.wsim_feature_flag:
            for base_i, target_i in self.pair_gen(self.stat.max_bef):
                self.WSIM_COLUMNS.append("WSIM_%d_%d" % (base_i, target_i))
            self.COLUMNS.extend(self.WSIM_COLUMNS)

            logging.debug("Creating SentenceWord2vecSimilarity object")
            self.wsim = SentenceWord2vecSimilarity(model_path, sim_func_type="mean_cos")
            logging.debug("Created SentenceWord2vecSimilarity object")

        if not any([self.standalone_feature_flag, self.keyword_feature_flag, self.tsim_feature_flag, self.wsim_feature_flag]):
            raise Exception("Use at least one feature type")

        self.name = self.COLUMNS

    def fit(self, dialogues):

        self.stat.fit(dialogues)

        # Keyword
        if self.keyword_feature_flag:
            self.of = {}
            for bef_i in range(self.stat.max_bef):
                self.of[bef_i] = create_of_from_stat(self.stat, bef_i)

            self.keywords = {}
            for bef_i in range(self.stat.max_bef):
                self.keywords[bef_i] = [t for t, ow in self.of[bef_i].most_high(n=self.keyword_n)]

        # TF similarity
        if self.tsim_feature_flag:
            self.tsim = SentenceTfidfSimilarity(self.stat["N"], self.stat["stemmed_df"])

    def extract(self, dialogue):
        features_list = []

        turns = dialogue["turns"]
        terms_list = [[t["stemmed"] for s in turn["tokens"] for t in s]
                      for turn in turns]

        for turn_i in range(len(turns)):
            features = []

            c_length = len(dialogue["turns"][turn_i]["utterance"])
            t_length = len(terms_list[turn_i])

            if self.standalone_feature_flag:
                features.extend([turn_i, c_length, t_length])

            if self.keyword_feature_flag:
                for bef_i in range(self.stat.max_bef):
                    target_turn_i = turn_i - bef_i
                    if target_turn_i >= 0:
                        features.extend([1 if k in terms_list[target_turn_i]
                                         else 0 for k in self.keywords[bef_i]])
                    else:
                        # features.extend([None for _ in range(len(keywords))])
                        features.extend([0 for _ in range(len(self.keywords[bef_i]))])

            if self.tsim_feature_flag:
                for base_i, target_i in self.pair_gen(self.stat.max_bef):
                    base_turn_i = turn_i - base_i
                    target_turn_i = turn_i - target_i
                    if base_turn_i >= 0 and target_turn_i >= 0:
                        tsim = self.tsim(terms_list[base_turn_i], terms_list[target_turn_i])
                        features.append(tsim)
                    else:
                        tsim = None  # Fill in mean value later
                        # tsim = 0.0
                        features.append(tsim)

            if self.wsim_feature_flag:
                for base_i, target_i in self.pair_gen(self.stat.max_bef):
                    base_turn_i = turn_i - base_i
                    target_turn_i = turn_i - target_i
                    if base_turn_i >= 0 and target_turn_i >= 0:
                        wsim = self.wsim(terms_list[base_turn_i], terms_list[target_turn_i])
                        features.append(wsim)
                    else:
                        wsim = None  # Fill in mean value later
                        # wsim = 0.0
                        features.append(wsim)

            features_list.append(features)

        return features_list

    def pair_gen(self, max_i):
        for base_i in range(max_i):
            for target_i in range(base_i + 1, max_i):
                yield base_i, target_i
