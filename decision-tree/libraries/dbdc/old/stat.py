# -*- coding: utf-8 -*-

import os
import logging
from collections import Counter, defaultdict


class StatInfo(dict):

    def __init__(self, tokenizer, max_bef=30):
        dict.__init__(self)
        self.tokenizer = tokenizer
        self.max_bef = max_bef

    def initialize(self):
        self["N"] = 0
        self["R"] = 0
        self.terms = []
        self.rterms = defaultdict(list)
        self.stemmed_terms = []
        self.stemmed_rterms = defaultdict(list)

    def fit(self, dialogues):

        for dialogue in dialogues:
            turns = dialogue.tokenized_turns(self.tokenizer)  # [{speaker:, terms:}]
            dist_list = dialogue.dist_list()

            # unique terms per turn for calculation of document (turn) frequency
            unique_terms_list = [list(set([t["token"] for s in turn["terms"] for t in s]))
                                 for turn in turns]
            unique_stemmed_terms_list = [list(set([t["stemmed"] for s in turn["terms"] for t in s]))
                                         for turn in turns]

            for turn_i in range(len(turns)):
                self["N"] += 1

                self.terms.extend(unique_terms_list[turn_i])
                self.stemmed_terms.extend(unique_stemmed_terms_list[turn_i])

                dist = dist_list[turn_i]
                if len(dist) == 3 and dist[0] < dist[2] and dist[1] < dist[2]:
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
