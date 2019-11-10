#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

from skato.nlp.utils import NGram, FloatCounter, CNG


class Estimator(dict):

    def __init__(self, mode="c", n=1, f=1.0, opt=None):
        dict.__init__(self)
        self.mode = mode
        self.n = n
        self.f = f
        self.ngram_func = NGram(n, mode=mode)

        if opt:
            self.optimize = getattr(self, "optimize_%s" % opt)
        else:
            self.optimize = self.optimize_default

    def fit(self, train_dialogues):
        train_A = defaultdict(dict)
        for d in train_dialogues:
            train_A[d["corpus"]][d["dialogue-id"]] = d

        self.mean = {}
        self.std = {}
        # self.one4th = {}
        # self.three4th = {}
        for corpus, A in train_A.items():
            self[corpus] = CNGDialogueVerifier(A, self.ngram_func, f=self.f)
            dissimilarities = []
            for target_dialogue_set in train_A.values():
                for dialogue in target_dialogue_set.values():
                    dissimilarities.append(self[corpus](dialogue))
            self.mean[corpus] = np.mean(dissimilarities)
            self.std[corpus] = np.std(dissimilarities)
            # self.one4th[corpus] = stats.scoreatpercentile(dissimilarities, 25)
            # self.three4th[corpus] = stats.scoreatpercentile(dissimilarities, 75)

        return self

    def optimize_default(self, score, corpus):
        return score

    def optimize_std(self, score, corpus):
        return (score - self.mean[corpus]) / self.std[corpus]

    # def optimize_box(self, score, corpus):
    #     return (score - self.one4th[corpus]) / (self.three4th[corpus] - self.one4th[corpus])

    def predict(self, target_dialogue):
        conf_scores = []
        for corpus in self.keys():
            conf_score = self.optimize(self[corpus](target_dialogue), corpus)
            conf_scores.append((corpus, conf_score))

        return sorted(conf_scores, key=lambda x: x[1])

    def score(self, test_dialogues):
        data = defaultdict(list)
        for dialogue in test_dialogues:
            data["GOLD"].append(dialogue["corpus"])
            sorted_conf_scores = self.predict(dialogue)
            pred = sorted_conf_scores[0][0]
            data["PRED"].append(pred)

            for corpus, conf in sorted_conf_scores:
                data["%s_CONF" % corpus].append(conf)

        df = pd.DataFrame(data)

        f_measures = []
        for corpus in self.keys():
            pred_corpus_idx = df["PRED"] == corpus
            gold_corpus_idx = df["GOLD"] == corpus
            correct_count = (pred_corpus_idx & gold_corpus_idx).sum()
            pred_count = pred_corpus_idx.sum()
            precision = correct_count / pred_count if pred_count > 0 else 0.0
            recall = correct_count / gold_corpus_idx.sum()
            f_measure = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0.0 else 0.0
            logging.info("%s: p=%f, r=%f, f=%f" % (corpus, precision, recall, f_measure))

            f_measures.append(f_measure)

        mean_f_measure = np.mean(f_measures)
        return mean_f_measure


class Params(dict):

    def __init__(self, mode, n, f, **kwargs):
        dict.__init__(self)
        self["mode"] = mode
        self["n"] = n
        self["f"] = f

    def __str__(self):
        return "{mode}-{n}-{f}".format(**self)

    def __hash__(self):
        return hash(str(self))


class CNGDialogueVerifier(object):

    def __init__(self, A, gram_func, f=1.0):
        self.gram_func = gram_func
        self.f = f
        self.ngf = {}
        self.D_max = defaultdict(float)

        for dialogue_id, dialogue in A.items():
            self.ngf[dialogue_id] = self.count_ngram(dialogue)

        self.calc()

    def __call__(self, dialogue, debug=False):
        return self.M(dialogue, debug=debug)

    def count_ngram(self, dialogue):
        ngf = FloatCounter()
        for turn in dialogue["turns"]:
            if turn["speaker"] == "S":
                ngf += FloatCounter(self.gram_func(turn["utterance"]))

        return ngf

    def calc(self):
        dialogue_ids = self.ngf.keys()

        self.D_matrix = []
        self.D_mean = []

        logging.debug("Calculating all pairs of %d dialogues" % len(dialogue_ids))
        for d_i_id in dialogue_ids:
            self.D_matrix.append([])
            for d_j_id in dialogue_ids:
                if d_i_id == d_j_id:
                    continue
                dis = CNG.dissimilarity(self.ngf[d_i_id], self.ngf[d_j_id], f=self.f)
                self.D_matrix[-1].append(dis)
                if self.D_max[d_i_id] < dis:
                    self.D_max[d_i_id] = dis
            self.D_mean.append(np.mean(self.D_matrix[-1]))
        logging.debug("Calculated all pairs of %d dialogues" % len(dialogue_ids))

    def M(self, target_dialogue, debug=False):
        dialogue_ids = self.ngf.keys()
        dis = []
        target_ngf = self.count_ngram(target_dialogue)

        # logging.debug("Calculating dis against a given dialogue")
        for d_i_id in dialogue_ids:
            dis.append(CNG.dissimilarity(self.ngf[d_i_id],
                                         target_ngf, f=self.f) / self.D_max[d_i_id])
        # logging.debug("Calculated dis against a given dialogue")

        if debug:
            verbose_strng = ["%s / %s" % (CNG.dissimilarity(self.ngf[d_i_id], target_ngf, f=0.75), self.D_max[d_i_id])
                             for d_i_id in dialogue_ids]

            # import ipdb
            # ipdb.set_trace()

        return np.average(dis)
