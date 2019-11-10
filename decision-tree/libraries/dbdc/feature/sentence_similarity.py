# -*- coding: utf-8 -*-

import os
import logging
import codecs
import json

from skato.nlp.utils import OfferWeight, SentenceWord2vecSimilarity, SentenceTfidfSimilarity

from .stat import StatInfo
from .abstract import AbstractExtractor


class SentenceSimilarityExtractor(AbstractExtractor):

    def pair_gen(self, max_i):
        for base_i in range(max_i):
            for target_i in range(base_i + 1, max_i):
                yield base_i, target_i


class SentenceWord2vecSimilarityExtractor(SentenceSimilarityExtractor):

    DOMAIN = "DIALOGUE"
    NEED_COLUMNS = ["tokens"]
    OUTPUT_COLUMNS = []
    COLUMN_FORMAT = "WSIM_%d_%d"
    SAVE_ATTRS = ["max_bef", "model_path", "sim_func_type"]

    def __init__(self, max_bef=None, sim_func_type="mean_cos", model_path=None):
        AbstractExtractor.__init__(self)
        self.max_bef = max_bef
        self.sim_func_type = sim_func_type
        self.model_path = model_path

    def initialize(self):
        if not self.max_bef:
            raise Exception("not self.max_bef")

        if not os.path.isfile(self.model_path):
            raise Exception("%s not found" % self.model_path)

        for base_i, target_i in self.pair_gen(self.max_bef):
            self.OUTPUT_COLUMNS.append(self.COLUMN_FORMAT % (base_i, target_i))

        logging.debug("Creating SentenceWord2vecSimilarity object")
        self.wsim = SentenceWord2vecSimilarity(self.model_path, sim_func_type=self.sim_func_type)
        logging.debug("Created SentenceWord2vecSimilarity object")

    def extract(self, dialogue):
        features_list = []

        terms_list = [[t["stemmed"] for s in turn["tokens"] for t in s]
                      for turn in dialogue["turns"]]

        for turn_i in range(len(terms_list)):
            features = []

            for base_i, target_i in self.pair_gen(self.max_bef):
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

    def output_path(self, save_dir):
        return os.path.join(save_dir, "%s.json" % self.__class__.__name__)

    def save(self, save_dir):
        output_path = self.output_path(save_dir)
        save_dict = {k: getattr(self, k) for k in self.SAVE_ATTRS}
        self.save_to_json(output_path, save_dict)

    def restore(self, save_dir):
        output_path = self.output_path(save_dir)
        load_dict = self.load_from_json(output_path)
        for k in self.SAVE_ATTRS:
            setattr(self, k, load_dict[k])

        self.initialize()


class SentenceTfidfSimilarityExtractor(SentenceSimilarityExtractor):

    DOMAIN = "DIALOGUE"
    NEED_COLUMNS = ["tokens"]
    OUTPUT_COLUMNS = []
    COLUMN_FORMAT = "TSIM_%d_%d"
    SAVE_ATTRS = ["max_bef", "N", "df_dict"]

    def __init__(self, max_bef=None):
        AbstractExtractor.__init__(self)
        self.max_bef = max_bef
        self.stat = None

    def fit(self, stat):
        if self.max_bef:
            if self.max_bef > stat.max_bef:
                raise Exception("self.max_bef > stat.max_bef")
        else:
            self.max_bef = stat.max_bef

        for base_i, target_i in self.pair_gen(self.max_bef):
            self.OUTPUT_COLUMNS.append(self.COLUMN_FORMAT % (base_i, target_i))

        self.tsim = SentenceTfidfSimilarity(stat["N"], stat["stemmed_df"])
        self.N = stat["N"]
        self.df_dict = stat["stemmed_df"]

    def extract(self, dialogue):
        features_list = []

        terms_list = [[t["stemmed"] for s in turn["tokens"] for t in s]
                      for turn in dialogue["turns"]]

        for turn_i in range(len(terms_list)):
            features = []

            for base_i, target_i in self.pair_gen(self.max_bef):
                base_turn_i = turn_i - base_i
                target_turn_i = turn_i - target_i
                if base_turn_i >= 0 and target_turn_i >= 0:
                    tsim = self.tsim(terms_list[base_turn_i], terms_list[target_turn_i])
                    features.append(tsim)
                else:
                    tsim = None  # Fill in mean value later
                    # tsim = 0.0
                    features.append(tsim)

            features_list.append(features)

        return features_list

    def output_path(self, save_dir):
        return os.path.join(save_dir, "%s.pkl" % self.__class__.__name__)

    def save(self, save_dir):
        output_path = self.output_path(save_dir)
        save_dict = {k: getattr(self, k) for k in self.SAVE_ATTRS}
        self.save_to_pkl(output_path, save_dict)

    def restore(self, save_dir):
        output_path = self.output_path(save_dir)
        load_dict = self.load_from_pkl(output_path)
        for k in self.SAVE_ATTRS:
            setattr(self, k, load_dict[k])

        for base_i, target_i in self.pair_gen(self.max_bef):
            self.OUTPUT_COLUMNS.append(self.COLUMN_FORMAT % (base_i, target_i))
        self.tsim = SentenceTfidfSimilarity(self.N, self.df_dict)
