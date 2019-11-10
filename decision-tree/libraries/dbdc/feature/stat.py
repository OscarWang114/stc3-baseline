# -*- coding: utf-8 -*-

import os
import logging
from collections import defaultdict, Counter


class StatInfo(dict):

    def __init__(self, max_bef=10):
        dict.__init__(self)

        if max_bef > 20:
            raise Exception("max_bef must be less than 20 (given max_bef is %d)" % max_bef)
        self.max_bef = max_bef

        self.initialize()

    def initialize(self):
        self["N"] = 0
        self["R"] = 0
        self.terms = []
        self.rterms = defaultdict(list)
        self.stemmed_terms = []
        self.stemmed_rterms = defaultdict(list)

    def fit(self, dialogues):

        for dialogue in dialogues:
            # unique terms per turn for calculation of document (turn) frequency
            turns = dialogue["turns"]
            tokens_list = [[token for s in turn["tokens"] for token in s]
                           for turn in turns]
            unique_terms_list = [set([token["token"] for token in tokens])
                                 for tokens in tokens_list]
            unique_stemmed_terms_list = [set([token["stemmed"] for token in tokens])
                                         for tokens in tokens_list]

            for turn_i in range(len(turns)):
                self["N"] += 1

                self.terms.extend(unique_terms_list[turn_i])
                self.stemmed_terms.extend(unique_stemmed_terms_list[turn_i])

                if turns[turn_i]["breakdown"] == "X":
                    self["R"] += 1

                    for bef_i in range(min(turn_i + 1, self.max_bef)):
                        target_turn_i = turn_i - bef_i
                        self.rterms[bef_i].extend(unique_terms_list[target_turn_i])
                        self.stemmed_rterms[bef_i].extend(unique_stemmed_terms_list[target_turn_i])

        self["df"] = Counter(self.terms)
        self["stemmed_df"] = Counter(self.stemmed_terms)

        self["rdf"] = {bef_i: Counter(self.rterms[bef_i])
                       for bef_i in self.rterms.keys()}
        self["stemmed_rdf"] = {bef_i: Counter(self.stemmed_rterms[bef_i])
                               for bef_i in self.stemmed_rterms.keys()}
