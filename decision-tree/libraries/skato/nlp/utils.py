# -*- coding: utf-8 -*-

import os
import logging
import math
import numpy as np
from collections import Counter
import jieba
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk.data
from krovetzstemmer import Stemmer


class ZhTokenizer(object):

    def tokenize(self, utterance):
        return jieba.lcut_for_search(utterance)


class EnTokenizer(object):

    def __init__(self,
                 s_tokenizer="tokenizers/punkt/english.pickle",
                 w_tokenizer="TreebankWordTokenizer",
                 stemmer="KrovetzStemmer", stemming=True):
        if s_tokenizer == "tokenizers/punkt/english.pickle":
            import nltk.data
            self.s_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        else:
            raise Exception("%s is not available" % s_tokenizer)

        if w_tokenizer == "TreebankWordTokenizer":
            from nltk.tokenize import TreebankWordTokenizer
            self.w_tokenizer = TreebankWordTokenizer()
        elif w_tokenizer == "WordPunctTokenizer":
            from nltk.tokenize import WordPunctTokenizer
            self.w_tokenizer = WordPunctTokenizer()
        else:
            raise Exception("%s is not available" % w_tokenizer)

        self.stemming = stemming
        if stemming:
            if isinstance(stemmer, str):
                if stemmer == "KrovetzStemmer":
                    from krovetzstemmer import Stemmer
                    self.stemmer = Stemmer()
                else:
                    raise Exception("%s is not available" % stemmer)
            else:
                self.stemmer = stemmer()

    def tokenize(self, text):
        sentences = self.s_tokenizer.tokenize(text)
        tokens_list = [self.w_tokenizer.tokenize(s) for s in sentences]
        if self.stemming:
            tokens_list = [[{"token": t, "stemmed": self.stemmer.stem(t)}
                            for t in s] for s in tokens_list]
        else:
            tokens_list = [[{"token": t} for t in s] for s in tokens_list]
        return tokens_list


class OfferWeight(dict):

    def __init__(self, N, R, n_dict, r_dict):
        dict.__init__(self)

        self.N = N
        self.R = R
        self.n_dict = n_dict
        self.r_dict = r_dict

        for t, r in r_dict.items():
            n = n_dict[t]  # must exist
            RW = math.log(((r + 0.5) * (N - n - R + r + 0.5)) / ((n - r + 0.5) * (R - r + 0.5)))
            self[t] = r * RW

    def __str__(self, con="\n"):
        return con.join(
            ["Term: %s, OW: %f, r: %d/%d, n: %d/%d" %
             (t, ow, self.r_dict[t], self.R, self.n_dict[t], self.N)
             for t, ow in self.most_high(n=10)])

    def most_high(self, n=None):
        sorted_items = sorted(self.items(), key=lambda x: x[1], reverse=True)

        if n:
            return sorted_items[:n]
        else:
            return sorted_items


class SentenceTfidfSimilarity(object):

    def __init__(self, N, n_dict, stop_words=None):
        self.N = N
        self.n_dict = n_dict

        if not stop_words:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words("english"))
        else:
            self.stop_words = stop_words

    def __call__(self, terms1, terms2):
        """only terms1 are iterated"""

        try:
            if len(terms1) == 0 or len(terms2) == 0:
                return 0.0
        except Exception as e:
            msg = "Terms 1: %s, Terms 2: %s" % (terms1, terms2)
            logging.warn(msg)
            raise Exception(msg)

        rel = 0.0
        s1tf = Counter(terms1)
        s2tf = Counter(terms2)
        for t, t1f in s1tf.items():
            if t in self.stop_words:
                continue

            t2f = s2tf.get(t, 0)
            if t2f == 0:
                continue

            r = math.log(t1f + 1) * \
                math.log(t2f + 1) * \
                math.log((self.N + 1) / (0.5 + self.n_dict[t]))

            rel += r

        return rel


class SentenceWord2vecSimilarity(object):

    sim_func_types = [
        "max_cos_and_geometric_mean",
        "mean_cos",
        "max_cos_and_arithmetic_mean",
    ]

    def __init__(self, model_path, sim_func_type="max_cos_and_geometric_mean"):
        self.sim = getattr(self, sim_func_type)

        from gensim.models import KeyedVectors
        self.model = KeyedVectors.load_word2vec_format(
            model_path, binary=True, unicode_errors="ignore")

    def __call__(self, terms1, terms2):
        # if terms2 == None:
        #     return None

        terms1, vectors1 = self.valid_terms_and_vectors(terms1)
        terms2, vectors2 = self.valid_terms_and_vectors(terms2)

        if len(terms1) == 0 or len(terms2) == 0:
            return 0.0
        else:
            return self.sim(vectors1, vectors2)

    def valid_terms_and_vectors(self, terms):
        valid_terms = []
        vectors = []
        for t in terms:
            if t in self.model:
                v = self.model[t]
                if np.all(np.isfinite(v)) and np.count_nonzero(v) != 0 and not np.isinf(np.linalg.norm(v)):
                    valid_terms.append(t)
                    vectors.append(v)
        return valid_terms, vectors

    def cos_matrix(self, vectors1, vectors2):
        absvectors1 = np.array([[np.linalg.norm(v) for v in vectors1]])
        absvectors2 = np.array([[np.linalg.norm(v) for v in vectors2]])
        #
        # for v in vectors1:
        #     if :
        #         print("INF")
        #         print(v)

        vectors1 = np.matrix(vectors1)
        vectors2 = np.matrix(vectors2)

        if not np.all(np.isfinite(absvectors1)):
            print("vectors1 not finite")
            print(absvectors1)
        if not np.all(np.isfinite(absvectors2)):
            print("vectors2 not finite")
        if np.count_nonzero(absvectors1) == 0:
            print("vectors1 all zeros")
        if np.count_nonzero(absvectors2) == 0:
            print("vectors2 all zeros")
        if len(absvectors1) == 0:
            print("vectors1 zero length")
        if len(absvectors2) == 0:
            print("vectors2 zero length")

        cos_matrix = (vectors1 * vectors2.T) / absvectors1.T / absvectors2
        return (cos_matrix + 1.0) / 2.0

    def max_cos_and_geometric_mean(self, vectors1, vectors2):
        cos_matrix = self.cos_matrix(vectors1, vectors2)
        coverage1 = np.max(cos_matrix, axis=1).mean()  # max
        coverage2 = np.max(cos_matrix, axis=0).mean()  # max
        return math.sqrt(coverage1 * coverage2)  # 幾何平均

    def mean_cos(self, vectors1, vectors2):
        cos_matrix = self.cos_matrix(vectors1, vectors2)
        return cos_matrix.mean()  # 平均

    def max_cos_and_arithmetic_mean(self, vectors1, vectors2):
        cos_matrix = self.cos_matrix(vectors1, vectors2)
        coverage1 = np.max(cos_matrix, axis=1).mean()  # max
        coverage2 = np.max(cos_matrix, axis=0).mean()  # max
        return (coverage1 + coverage2) / 2  # 算術平均


class EnsemblesNGram(object):

    def __init__(self, c_n, w_n):
        self.gram = {}
        if len(c_n) > 0:
            self.gram["c"] = {n: NGram(n, mode="c") for n in c_n}
        if len(w_n) > 0:
            self.gram["w"] = {n: NGram(n, mode="w") for n in w_n}

    def __call__(self, text):
        grams = []
        for mode, ngrams in self.gram.items():
            for n, ngram in ngrams.items():
                grams.extend(ngram(text))
        return grams


class NGram(object):

    def __init__(self, n, mode="c", lang="en"):
        self.n = n
        if mode == "c":
            self.set_char_tokenizer()
        elif mode == "w":
            if lang == "en":
                self.set_en_word_tokenizer()

    def __call__(self, text):
        tokens = self.tokenize(text)
        ngram = []

        for i in range(len(tokens)):
            cw = []

            if i >= self.n - 1:
                for j in reversed(range(self.n)):
                    cw.append(tokens[i - j])
            else:
                continue

            ngram.append(tuple(cw))

        return ngram

    def set_char_tokenizer(self):
        self.tokenize = lambda text: [c for c in text]

    def set_en_word_tokenizer(self):
        self.tokenizer = Tokenizer()
        self.tokenize = self.word_tokenize

    def word_tokenize(self, text):
        return [t["token"] for s in self.tokenizer.tokenize(text) for t in s]


class CNG(object):

    @staticmethod
    def dissimilarity(f_P1, f_P2, f=1.0):
        f_P1_keys = sorted(f_P1.items(), key=lambda x: x[1], reverse=True)
        f_P1_keys = [t[0] for t in f_P1_keys[:int(len(f_P1_keys) * f)]]
        f_P2_keys = sorted(f_P2.items(), key=lambda x: x[1], reverse=True)
        f_P2_keys = [t[0] for t in f_P2_keys[:int(len(f_P2_keys) * f)]]
        commons = set(f_P1_keys) | set(f_P2_keys)
        dis = 0.0
        for x in commons:
            f_P1_x = f_P1.get(x, 0.0)
            f_P2_x = f_P2.get(x, 0.0)
            try:
                dis += ((f_P1_x - f_P2_x) / ((f_P1_x + f_P2_x) / 2)) ** 2
            except Exception as e:
                import ipdb
                ipdb.set_trace()
        return dis


class FloatCounter(dict):

    def __init__(self, tokens=[]):
        l = float(len(tokens))
        if l > 0:
            count = Counter(tokens)
            self.update({k: v / l for k, v in count.items()})
        else:
            dict.__init__(self)

    def __add__(self, x):
        y = FloatCounter()
        keys = set(self.keys()) | set(x.keys())
        y.update({k: (self.get(k, 0.0) + x.get(k, 0.0)) / 2.0 for k in keys})
        return y
