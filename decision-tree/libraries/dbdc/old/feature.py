# -*- coding: utf-8 -*-

import os
import logging
import glob
import json
import math
from collections import Counter, defaultdict

from skato.nlp.utils import OfferWeight, SentenceTfidfSimilarity, SentenceWord2vecSimilarity

from .common import Dialogue
from .stat import StatInfo


def create_of_from_stat(stat, bef_i):
    return OfferWeight(stat["N"], stat["R"], stat["stemmed_df"], stat["stemmed_rdf"][bef_i])


class FeatureExtractor(object):

    def __init__(self, tokenizer, model_path, max_bef=3, keyword_n=10, **kwargs):
        self.stat = StatInfo(tokenizer, max_bef=max_bef)
        self.stat.initialize()
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

        turns = dialogue.tokenized_turns(self.stat.tokenizer)  # [{speaker:, terms:}]
        terms_list = [[t["stemmed"] for s in turn["terms"] for t in s]
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


class Features(object):

    tsim = None
    wsim = None

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __iter__(self):
        for corpus, dialogue in self.dialogues:
            lu_terms = None
            ls_terms = None

            for turn in dialogue["turns"]:

                u = tokenizer.tokenize(turn["utterance"])
                terms = [t["stemmed"] for s in u for t in s]  # 重複あり

                if turn["speaker"] == "S" and turn["turn-index"] == 0:
                    ls_terms = terms
                elif turn["speaker"] == "S" and turn["turn-index"] > 0:
                    c_length = len(turn["utterance"])
                    t_length = len(terms)
                    features = [turn["turn-index"], c_length, t_length]

                    # Calc tfidf similarity
                    tsim_cs_lu = tsim(terms, lu_terms)
                    if tsim_cs_lu is None:
                        logging.warning("Rel(lu|cs) is None, %s, %s, %d" %
                                        (corpus, dialogue["dialogue-id"], turn["turn-index"]))
                    tsim_cs_ls = tsim(terms, ls_terms)
                    if tsim_cs_ls is None:
                        logging.warning("Rel(ls|cs) is None, %s, %s, %d" %
                                        (corpus, dialogue["dialogue-id"], turn["turn-index"]))
                    tsim_lu_ls = tsim(lu_terms, ls_terms)
                    if tsim_lu_ls is None:
                        logging.warning("Rel(ls|lu) is None, %s, %s, %d" %
                                        (corpus, dialogue["dialogue-id"], turn["turn-index"]))
                    features.extend([tsim_cs_lu, tsim_cs_ls, tsim_lu_ls])

                    # Calc word2vec similarity
                    wsim_cs_lu = wsim(terms, lu_terms)
                    if wsim_cs_lu == 0.0:
                        logging.warning("Sim(cs,lu) == 0.0, %s, %s, %d" %
                                        (corpus, dialogue["dialogue-id"], turn["turn-index"]))
                    wsim_cs_ls = wsim(terms, ls_terms)
                    if wsim_cs_ls == 0.0:
                        logging.warning("Sim(cs,ls) == 0.0, %s, %s, %d" %
                                        (corpus, dialogue["dialogue-id"], turn["turn-index"]))
                    wsim_lu_ls = wsim(lu_terms, ls_terms)
                    if wsim_lu_ls == 0.0:
                        logging.warning("Sim(lu,ls) == 0.0, %s, %s, %d" %
                                        (corpus, dialogue["dialogue-id"], turn["turn-index"]))
                    features.extend([wsim_cs_lu, wsim_cs_ls, wsim_lu_ls])

                    # Add keyword features
                    features.extend(
                        Dialogue.get_keyword_features(keywords, terms))

                    eX.append(features)
                    x_turn_indexes.append(turn["turn-index"])

                    ls_terms = terms

                elif turn["speaker"] == "U":
                    lu_terms = terms
