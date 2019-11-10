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
from skato.nlp.utils import EnTokenizer, JaTokenizer
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
from dbdc.feature.preprocessor import Preprocessor
from dbdc.distribution.converter import (
    DistributionConverter,
    StatDistributionConverter,
)
from dbdc.predictor.rf import SimpleRFPredictor


def predict_a_global_model(**kwargs):

    logging.info(kwargs)

    # Load data
    corpora = Corpus.list(lang=kwargs["lang"])
    test_corpus_dialogues = load_dialogues(
        os.path.join(kwargs["eval_dir"], kwargs["lang"]), corpora)
    logging.debug({k: len(v) for k, v in test_corpus_dialogues.items()})
    global_dialogues = [d for dialogues in test_corpus_dialogues.values()
                        for d in dialogues]

    if kwargs["lang"] == "en" or kwargs["lang"] == "en.unrevised" or kwargs["lang"] == "en4":
        tokenizer = EnTokenizer(stemmer=StemmerExceptC)
    elif kwargs["lang"] == "jp" or kwargs["lang"] == "jp4" or kwargs["lang"] == "jp4_compe":
        tokenizer = JaTokenizer()
    else:
        raise Exception("%s is invalid language" % kwargs["lang"])

    # Prepare
    # stat_info = StatInfo(max_bef=kwargs["max_bef"])
    basic_extractors = [
        DistributionExtractor(),
        ProbabilityExtractor(),
        BreakdownLabelExtractor(),
    ]

    # Feature
    feature_flags = {
        "S": True,
        "K": True,
        "T": True,
        "W": True,
    }
    feature_repr = "".join([k for k, v in feature_flags.items() if v])
    feature_extractors = [
        TurnLengthExtractor(),
        KeywordExtractor(max_bef=kwargs["max_bef"],
                         keyword_n=kwargs["keyword_n"]),
        SentenceTfidfSimilarityExtractor(max_bef=kwargs["max_bef"]),
        SentenceWord2vecSimilarityExtractor(max_bef=kwargs["max_bef"],
                                            model_path=kwargs["model_path"]),
    ]

    global_dir = os.path.join(kwargs["model_dir"], kwargs["model_type"], feature_repr)
    extractor_dir = os.path.join(global_dir, "extractors")

    # have to load a word2vec model before tokenizing a text
    # which includes UnicodeEncodeError
    for extr_i in range(len(feature_extractors)):
        feature_extractors[extr_i].restore(extractor_dir)

    filler = NAFiller()
    filler.restore(extractor_dir)

    feature_columns = ["TURN-INDEX"] + [col.upper()
                                        for extr in feature_extractors
                                        for col in extr.OUTPUT_COLUMNS]

    preprocessor = Preprocessor(feature_columns, feature_extractors, filler)

    # RF
    predictor_dir = os.path.join(global_dir, "predictors")
    convs = {
        "OTX": DistributionConverter(),
        "AV": StatDistributionConverter(),
    }

    global_predictors = {
        feature_repr: {conv_type: SimpleRFPredictor(preprocessor, conv)
                       for conv_type, conv in convs.items()}
    }

    predictor_dir = os.path.join(global_dir, "predictors")
    for conv_type in global_predictors[feature_repr].keys():
        for model_i in range(1):
            model_dir = os.path.join(predictor_dir, conv_type, "%03d" % model_i)
            logging.info("Restoring a model from %s" % model_dir)
            global_predictors[feature_repr][conv_type].restore(model_dir)
            logging.info("Restored a model from %s" % model_dir)

    # Predict
    for d_i in range(len(global_dialogues)):
        global_dialogues[d_i].add_tokens(tokenizer)
        global_dialogues[d_i].add_features(basic_extractors)

    result_filename_format = "{model_type}-{feature_type}-{conv_type}-{result_number:03d}.csv"
    for conv_type in global_predictors[feature_repr].keys():

        result_type = {
            "model_type": kwargs["model_type"],
            "feature_type": feature_repr,
            "conv_type": conv_type,
            "result_number": 0,
        }

        logging.info("Predicting %d dialogues by %s" % (len(global_dialogues), result_type))
        res = []
        for d_i in range(len(global_dialogues)):
            pred_y = global_predictors[feature_repr][conv_type].predict(global_dialogues[d_i])
            res.append(pred_y)
        res = pd.concat(res, axis=0, sort=False).reset_index(drop=True)
        logging.info("Predicted %d dialogues by %s" % (len(global_dialogues), result_type))
        output_path = os.path.join(
            kwargs["output_dir"], result_filename_format.format(**result_type))
        logging.info("Saving a result to %s" % output_path)
        res.to_csv(output_path, index=False)
        logging.info("Savied a result to %s" % output_path)

    return


if __name__ == "__main__":
    main(__file__, globals())
