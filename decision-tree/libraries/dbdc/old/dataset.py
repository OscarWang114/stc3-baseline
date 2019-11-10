# -*- coding: utf-8 -*-


import logging


class Corpus(object):

    @staticmethod
    def list(lang="en"):
        if lang == "en":
            return ["CIC", "YI", "IRIS", "TickTock"]
        elif lang == "jp":
            return ["DCM", "DIT", "IRS"]
        else:
            raise Exception("%s is invalid language" % lang)
