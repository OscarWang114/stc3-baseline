# -*- coding: utf-8 -*-


import logging
import krovetzstemmer


class StemmerExceptC():

    def __init__(self):
        self.stemmer = krovetzstemmer.Stemmer()

    def stem(self, term):
        if term != "C":
            return self.stemmer.stem(term)
        else:
            logging.info("Stem C without krovetzstemmer module")
            return "c"
