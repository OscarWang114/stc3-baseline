# -*- coding: utf-8 -*-

import os
import logging
import codecs
import json


from .abstract import AbstractExtractor


class TurnLengthExtractor(AbstractExtractor):

    DOMAIN = "TURN"
    NEED_COLUMNS = ["tokens", "utterance"]
    OUTPUT_COLUMNS = ["T_LENGTH", "C_LENGTH"]

    def extract(self, turn):
        tokens = [token["token"]  # not stemmed tokens
                  for sent in turn["tokens"] for token in sent]
        return [len(tokens), len(turn["utterance"])]
