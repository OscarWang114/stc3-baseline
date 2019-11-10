# -*- coding: utf-8 -*-

import os
import logging
import codecs
import json

from skato.nlp.utils import OfferWeight

from .stat import StatInfo
from .abstract import AbstractExtractor


class KeywordExtractor(AbstractExtractor):

    DOMAIN = "DIALOGUE"
    NEED_COLUMNS = ["tokens"]
    OUTPUT_COLUMNS = []
    COLUMN_FORMAT = "KEYWORD_%02d_IN_%d"
    SAVE_ATTRS = ["max_bef", "keyword_n", "keywords", "col2key"]

    def __init__(self, max_bef=None, keyword_n=10):
        AbstractExtractor.__init__(self)
        self.max_bef = max_bef
        self.keyword_n = keyword_n
        self.of = {}
        self.keywords = {}
        self.col2key = {}
        self.stat = None

    def fit(self, stat):
        if self.max_bef:
            if self.max_bef > stat.max_bef:
                raise Exception("self.max_bef > stat.max_bef")
        else:
            self.max_bef = stat.max_bef

        # Select keywords
        for bef_i in range(self.max_bef):
            self.of[bef_i] = OfferWeight(
                stat["N"], stat["R"], stat["stemmed_df"], stat["stemmed_rdf"][bef_i])
            self.keywords[bef_i] = [t for t, ow in self.of[bef_i].most_high(n=self.keyword_n)]

            # Set column names
            for keyword_i, keyword in enumerate(self.keywords[bef_i]):
                col = self.COLUMN_FORMAT % (keyword_i, bef_i)
                self.OUTPUT_COLUMNS.append("KEYWORD_%02d_IN_%d" % (keyword_i, bef_i))
                self.col2key[col] = keyword

        self.stat = stat

    # def extract(self, dialogue):
    #     features_list = []
    #
    #     terms_list = [[t["stemmed"] for s in turn["tokens"] for t in s]
    #                   for turn in dialogue["turns"]]
    #
    #     for turn_i in range(len(terms_list)):
    #         features = []
    #
    #         for bef_i in range(self.max_bef):
    #             target_turn_i = turn_i - bef_i
    #             if target_turn_i >= 0:
    #                 for keyword_i, keyword in enumerate(self.keywords[bef_i]):
    #                     keyword_flag = 1 if keyword in terms_list[target_turn_i] else 0
    #                     features.append(keyword_flag)
    #             else:
    #                 # features.extend([None for _ in range(len(keywords))])
    #                 features.extend([0 for _ in range(self.keyword_n)])
    #
    #         features_list.append(features)
    #
    #     return features_list

    def extract(self, dialogue):
        features_list = []

        terms_list = [[t["stemmed"] for s in turn["tokens"] for t in s]
                      for turn in dialogue["turns"]]

        for turn_i in range(len(terms_list)):
            features = []

            for bef_i in range(self.max_bef):
                #                 target_turn_i = turn_i - bef_i
                target_turn_i = turn_i  # 常に target_turn_i >=0
                if target_turn_i >= 0:
                    for keyword_i, keyword in enumerate(self.keywords[bef_i]):
                        keyword_flag = 1 if keyword in terms_list[target_turn_i] else 0
                        features.append(keyword_flag)
                else:
                    # features.extend([None for _ in range(len(keywords))])
                    features.extend([0 for _ in range(self.keyword_n)])

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

        _keywords = {int(k): v for k, v in self.keywords.items()}
        self.keywords = _keywords

        # Set column names
        for bef_i in range(self.max_bef):
            for keyword_i, keyword in enumerate(self.keywords[bef_i]):
                col = self.COLUMN_FORMAT % (keyword_i, bef_i)
                self.OUTPUT_COLUMNS.append("KEYWORD_%02d_IN_%d" % (keyword_i, bef_i))
                # self.col2key[col] = keyword
