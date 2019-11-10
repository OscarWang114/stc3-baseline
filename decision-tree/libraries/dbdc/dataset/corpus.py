# -*- coding: utf-8 -*-

import logging


class Corpus(object):

    @staticmethod
    def list(lang="en"):
        if lang == "en" or lang == "zh":
            return ["C", "D", "L", "O"]
        else:
            raise Exception("%s is invalid language" % lang)


class Corpora(list):
    pass
