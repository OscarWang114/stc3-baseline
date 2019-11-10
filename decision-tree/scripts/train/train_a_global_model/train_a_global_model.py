#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import logging
import datetime
import math
import copy
import codecs
import pickle
import glob
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd


from skato.exec.utils import main
from dbdc.dataset.corpus import Corpus
from dbdc.base.json import Jsons
from dbdc.base.dialogue import Dialogues, load_dialogues
from skato.nlp.utils import ZhTokenizer, EnTokenizer
from skato.nlp.stemmer import StemmerExceptC
from dbdc.feature.stat import StatInfo
from dbdc.feature.standalone import TurnLengthExtractor
from dbdc.feature.keyword import KeywordExtractor
from dbdc.feature.sentence_similarity import (
    SentenceTfidfSimilarityExtractor,
    SentenceWord2vecSimilarityExtractor,
)
from dbdc.feature.extractors import (
    DistributionExtractor,
    ProbabilityExtractor,
    BreakdownLabelExtractor,
)
from dbdc.feature.fillna import NAFiller
from dbdc.distribution.converter import (
    DistributionConverter,
    StatDistributionConverter,
)
from dbdc.predictor.rf import RFManager

def train_a_global_model(**kwargs):
    logging.info(kwargs)

    corpora = Corpus.list(lang=kwargs["lang"])
    train_corpus_dialogues = load_dialogues(
        os.path.join(kwargs["dev_dir"], kwargs["lang"]), corpora)
    logging.debug({k: len(v) for k, v in train_corpus_dialogues.items()})

    if kwargs["lang"] == "en":
        tokenizer = EnTokenizer(stemmer=StemmerExceptC)
    elif kwargs["lang"] == "zh":
        tokenizer = ZhTokenizer()
    else:
        raise Exception("%s is invalid language" % kwargs["lang"])

    return

    stat_info = StatInfo(max_bef=kwargs["max_bef"])
    basic_extractors = [
        DistributionExtractor(),
        ProbabilityExtractor(),
        BreakdownLabelExtractor(),
    ]
    feature_extractors = [
        TurnLengthExtractor(),
        KeywordExtractor(max_bef=kwargs["max_bef"],
                         keyword_n=kwargs["keyword_n"]),
        SentenceTfidfSimilarityExtractor(max_bef=kwargs["max_bef"]),
    ]
    wsim_extr = SentenceWord2vecSimilarityExtractor(max_bef=kwargs["max_bef"],
                                                    model_path=kwargs["model_path"])

    # have to load a word2vec model before tokenizing a text
    # which includes UnicodeEncodeError
    wsim_extr.initialize()
    feature_extractors.append(wsim_extr)

    global_dialogues = [d for dialogues in train_corpus_dialogues.values()
                        for d in dialogues]

    for d_i in range(len(global_dialogues)):
        global_dialogues[d_i].add_tokens(tokenizer)
        global_dialogues[d_i].add_features(basic_extractors)

    stat_info.initialize()
    stat_info.fit(global_dialogues)
    logging.debug((stat_info["N"], stat_info["R"]))

    for extr_i in range(len(feature_extractors)):
        # have to load a word2vec model before tokenizing a text
        # which includes UnicodeEncodeError
        if feature_extractors[extr_i].__class__ != SentenceWord2vecSimilarityExtractor:
            feature_extractors[extr_i].initialize()
            feature_extractors[extr_i].fit(stat_info)

    for d_i in range(len(global_dialogues)):
        global_dialogues[d_i].add_features(feature_extractors)

    global_df = []
    for d_i in range(len(global_dialogues)):
        global_df.append(global_dialogues[d_i].to_df())
    global_df = pd.concat(global_df, axis=0, sort=False).reset_index(drop=True)
    logging.debug("len(global_df) = %d" % len(global_df))

    global_eval_df = global_df[
        np.array(global_df["EVAL_INDEX"]) >= 0].reset_index(drop=True)
    logging.debug("len(global_eval_df) = %d" % len(global_eval_df))

    feature_columns = ["TURN-INDEX"] + [col.upper()
                                        for extr in feature_extractors
                                        for col in extr.OUTPUT_COLUMNS]

    logging.debug("feature columns = %s" % feature_columns)

    X = global_eval_df[feature_columns]

    filler = NAFiller()
    filler.fit(X)
    logging.debug("NA columns = %s" % filler.X_columns)

    X = filler.apply(X)
    logging.debug("X.isna().any(axis=0).any() = %s" %
                  X.isna().any(axis=0).any())

    convs = {
        "OTX": DistributionConverter(),
        "AV": StatDistributionConverter(),
    }

    feature_flags = {
        "S": True,
        "K": True,
        "T": True,
        "W": True,
    }
    feature_repr = "".join([k for k, v in feature_flags.items() if v])

    global_rfs = {
        feature_repr: {conv_type: RFManager(conv.TARGET_COLUMNS)
                       for conv_type, conv in convs.items()}
    }

    Y = global_eval_df[convs["OTX"].SOURCE_COLUMNS]

    for feature_type in global_rfs.keys():
        for conv_type in global_rfs[feature_type].keys():
            global_rfs[feature_type][conv_type].train(
                X, convs[conv_type].convert(Y))

    for feature_type in global_rfs.keys():
        feature_dir = os.path.join(kwargs["save_dir"], "global", feature_type)

        extractor_dir = os.path.join(feature_dir, "extractors")
        for extr in feature_extractors:
            extr.save(extractor_dir)
        filler.save(extractor_dir)

        predictor_dir = os.path.join(feature_dir, "predictors")
        for conv_type in global_rfs[feature_type].keys():
            for model_i in range(1):
                model_dir = os.path.join(
                    predictor_dir, conv_type, "%03d" % model_i)
                global_rfs[feature_type][conv_type].save(model_dir)

    # Each Corpus

    for corpus in train_corpus_dialogues.keys():

        stat_info = StatInfo(max_bef=kwargs["max_bef"])
        stat_info.initialize()
        stat_info.fit(train_corpus_dialogues[corpus])
        logging.debug((corpus, stat_info["N"], stat_info["R"]))

        feature_extractors = [
            TurnLengthExtractor(),
            KeywordExtractor(max_bef=kwargs["max_bef"],
                             keyword_n=kwargs["keyword_n"]),
            SentenceTfidfSimilarityExtractor(max_bef=kwargs["max_bef"]),
        ]
        for extr_i in range(len(feature_extractors)):
            feature_extractors[extr_i].initialize()
            feature_extractors[extr_i].fit(stat_info)

        # Add SentenceWord2vecSimilarityExtractor
        feature_extractors.append(wsim_extr)

        for d_i in range(len(global_dialogues)):
            global_dialogues[d_i].add_features(feature_extractors)

        corpus_df = []
        for d_i in range(len(train_corpus_dialogues[corpus])):
            corpus_df.append(train_corpus_dialogues[corpus][d_i].to_df())
        corpus_df = pd.concat(
            corpus_df, axis=0, sort=False).reset_index(drop=True)
        logging.debug((corpus, "len(corpus_df) = %d" % len(corpus_df)))

        corpus_eval_df = corpus_df[
            np.array(corpus_df["EVAL_INDEX"]) >= 0].reset_index(drop=True)
        logging.debug((corpus, "len(corpus_eval_df) = %d" %
                       len(corpus_eval_df)))

        X = corpus_eval_df[feature_columns]

        corpus_filler = NAFiller()
        corpus_filler.fit(X)
        logging.debug("NA columns = %s" % corpus_filler.X_columns)

        X = corpus_filler.apply(X)
        logging.debug("X.isna().any(axis=0).any() = %s" %
                      X.isna().any(axis=0).any())

        convs = {
            "OTX": DistributionConverter(),
            "AV": StatDistributionConverter(),
        }

        corpus_rfs = {
            feature_repr: {conv_type: RFManager(conv.TARGET_COLUMNS)
                           for conv_type, conv in convs.items()}
        }

        Y = corpus_eval_df[convs["OTX"].SOURCE_COLUMNS]

        for feature_type in corpus_rfs.keys():
            for conv_type in corpus_rfs[feature_type].keys():
                corpus_rfs[feature_type][conv_type].train(
                    X, convs[conv_type].convert(Y))

        for feature_type in corpus_rfs.keys():
            feature_dir = os.path.join(
                kwargs["save_dir"], corpus, feature_type)

            extractor_dir = os.path.join(feature_dir, "extractors")
            for extr in feature_extractors:
                extr.save(extractor_dir)
            corpus_filler.save(extractor_dir)

            predictor_dir = os.path.join(feature_dir, "predictors")
            for conv_type in corpus_rfs[feature_type].keys():
                for model_i in range(1):
                    model_dir = os.path.join(
                        predictor_dir, conv_type, "%03d" % model_i)
                    corpus_rfs[feature_type][conv_type].save(model_dir)


if __name__ == "__main__":
    main(__file__, globals())
