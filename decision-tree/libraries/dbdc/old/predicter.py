# -*- coding: utf-8 -*-

import os
import logging
import json
import codecs
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor


from .verifier import Estimator, Params


class AbstractBreakdownPredictor(object):

    LABELED_JSON_FORMAT = "{did}.labels.json"

    def fit(self):
        raise Exception("fit function is not implemented")

    def predict(self):
        raise Exception("predict function is not implemented")

    def label(self, test_dialogue):
        labeled_dialogue = {"dialogue-id": test_dialogue["dialogue-id"], "turns": []}

        turn_index_list = []
        for turn in test_dialogue["turns"]:
            if turn['speaker'] == "U" or turn['annotations'] == []:
                continue
            else:
                turn_index_list.append(turn["turn-index"])

        Y = self.predict(test_dialogue)

        # import ipdb
        # ipdb.set_trace()

        if len(Y) != len(turn_index_list):
            raise Exception('len(Y) != len(turn_index_list)')

        for turn_index, y in zip(turn_index_list, Y):
            labeled_dialogue["turns"].append({
                "turn-index": turn_index,
                "labels": [{
                    "breakdown": ["O", "T", "X"][np.argmax(y)],
                    "prob-O": y[0],
                    "prob-T": y[1],
                    "prob-X": y[2],
                }]
            })

        return labeled_dialogue

    def predict_ans_save(self, output_dir, test_dialogues):
        for test_dialogue in test_dialogues:
            dialogue_id = test_dialogue["dialogue-id"]
            labeled_dialogue = self.label(test_dialogue)
            output_filename = self.LABELED_JSON_FORMAT.format(did=dialogue_id)
            output_path = os.path.join(output_dir, output_filename)
            json.dump(labeled_dialogue, codecs.open(output_path, "w", "utf-8"), indent=2)

    # def score(self, test_dialogues):
    #     scores = []
    #     for test_dialogue in test_dialogues:
    #         dialogue_id = test_dialogue["dialogue-id"]
    #         labeled_dialogue = self.predict(test_dialogue)
    #         import ipdb
    #         ipdb.set_trace()
    #         score = Dialogue.score(labeled_dialogue, test_dialogue)
    #         scores.append(score)
    #     mean_score = np.mean(scores)
    #     return mean_score


class CNGDissimilarityBasedETRBreakdownPredictor(AbstractBreakdownPredictor):

    def __init__(self, known_corpus_list, extractor, converter, opt=None, **kwargs):
        self.corpora = known_corpus_list
        self.predictors = {corpus: ETRBreakdownPredictor(extractor, converter, **kwargs)
                           for corpus in known_corpus_list}

        params = Params("c", 2, 1.0)

        if opt == "conf_weight":
            opt = "default"
            self.predict = self.predict_conf_weight
            self.global_predictor = ETRBreakdownPredictor(extractor, converter, **kwargs)
        elif opt == "conf_threshold":
            opt = "default"
            self.predict = self.predict_conf_threshold
            self.global_predictor = ETRBreakdownPredictor(extractor, converter, **kwargs)
        else:
            self.global_predictor = False

        self.estimator = Estimator(opt=opt, **params)

    def fit(self, train_dialogues):
        self.estimator.fit(train_dialogues)
        if self.global_predictor:
            self.global_predictor.fit(train_dialogues)

        train_dialogue_list = defaultdict(list)
        for dialogue in train_dialogues:
            train_dialogue_list[dialogue["corpus"]].append(dialogue)

        if len(train_dialogue_list.keys()) != len(self.corpora):
            logging.warn("len(train_dialogue_list.keys()) != len(self.corpora)")

        for corpus in self.corpora:
            dialogues = train_dialogue_list[corpus]
            self.predictors[corpus].fit(dialogues)

    def predict(self, test_dialogue):
        conf_scores = self.estimator.predict(test_dialogue)
        estimated_corpus = conf_scores[0][0]
        predicted_dist_list = self.predictors[estimated_corpus].predict(test_dialogue)
        return predicted_dist_list

    def predict_conf_weight(self, test_dialogue):
        conf_scores = self.estimator.predict(test_dialogue)
        conf_weights = [(t[0], 1.0 - t[1]) for t in conf_scores if t[1] < 1.0]
        if len(conf_weights) < 1:
            logging.warn("len(conf_weights) < 1 @%s" % test_dialogue["dialogue-id"])
            predicted_dist_list = self.global_predictor.predict(test_dialogue)
        else:
            sum_weight = sum([t[1] for t in conf_weights])
            normalized_conf_weights = [(t[0], t[1] / sum_weight) for t in conf_weights]

            predicted_dist_list = None
            for corpus, conf_weight in conf_weights:
                dist_list = self.predictors[corpus].predict(test_dialogue)
                if not isinstance(dist_list, np.ndarray):
                    predicted_dist_list = np.array(dist_list) * conf_weight
                else:
                    predicted_dist_list += np.array(dist_list) * conf_weight
        return predicted_dist_list

    def predict_conf_threshold(self, test_dialogue):
        conf_scores = self.estimator.predict(test_dialogue)
        filtered_corpora = [t[0] for t in conf_scores if t[1] < 1.0]
        if len(filtered_corpora) < 1:
            logging.warn("len(filtered_corpora) < 1 @%s" % test_dialogue["dialogue-id"])
            predicted_dist_list = self.global_predictor.predict(test_dialogue)
        else:
            sum_n = float(len(filtered_corpora))
            predicted_dist_list = None
            for corpus in filtered_corpora:
                dist_list = self.predictors[corpus].predict(test_dialogue)
                if not isinstance(dist_list, np.ndarray):
                    predicted_dist_list = np.array(dist_list) / sum_n
                else:
                    predicted_dist_list += np.array(dist_list) / sum_n
        return predicted_dist_list

    def save(self, save_dir):
        for corpus in self.predictors.keys():
            corpus_dir = os.path.join(save_dir, corpus)
            if not os.path.isdir(corpus_dir):
                os.mkdir(corpus_dir)

            self.predictors[corpus].save(corpus_dir)

    def restore(self, save_dir):
        for corpus in self.predictors.keys():
            corpus_dir = os.path.join(save_dir, corpus)
            if not os.path.isdir(corpus_dir):
                raise Exception("%s is not found" % corpus_dir)

            self.predictors[corpus].restore(corpus_dir)


class ETRBreakdownPredictor(AbstractBreakdownPredictor):

    DEFAULT_ETR_PARAMS = {
        "n_estimators": 10,
        "criterion": "mse",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "auto",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "min_impurity_split": None,
        "bootstrap": False,
        "oob_score": False,
        "n_jobs": 1,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
    }

    def __init__(self, extractor, converter, **kwargs):
        self.extr = extractor
        self.conv = converter

        self.etr_params = {k: kwargs.get(k, self.DEFAULT_ETR_PARAMS[k])
                           for k in self.DEFAULT_ETR_PARAMS.keys()}

    def fit(self, train_dialogues):
        self.extr.fit(train_dialogues)

        X = []
        Y = []
        for dialogue in train_dialogues:
            features_list = self.extr.extract(dialogue)
            dist_list = dialogue.dist_list()

            for features, dist in zip(features_list, dist_list):
                if len(dist) > 0:  # only annotated turns
                    targets = self.conv.convert(dist)
                    X.append(features)
                    Y.append(targets)

        X = pd.DataFrame(X, columns=self.extr.COLUMNS)
        Y = pd.DataFrame(Y, columns=self.conv.COLUMNS)

        try:
            if self.extr.tsim_feature_flag and self.extr.standalone_feature_flag:
                # Check null row
                tsim12_null_n = len(X[X["TSIM_1_2"].isnull()])
                if tsim12_null_n > 0:
                    turn_i_sum = X[X["TSIM_1_2"].isnull()]["TURN_I"].sum()
                    if not (turn_i_sum == tsim12_null_n or turn_i_sum == 0):
                        # tsim が nulll になっているのは最初の発話のみか確認
                        # X.loc[X.isnull().any(axis=1), :]
                        import ipdb
                        ipdb.set_trace()
                        raise Exception(
                            'X[X["TSIM_1_2"].isnull()]["TURN_I"].sum() != len(X[X["TSIM_1_2"].isnull()])')
        except Exception as e:
            import ipdb
            ipdb.set_trace()
            raise e

        self.X_mean = X.mean()
        for col in (self.extr.TSIM_COLUMNS + self.extr.WSIM_COLUMNS):
            X[col].fillna(self.X_mean[col], inplace=True)

        # Convert dtypes
        for col in (self.extr.STANDALONE_COLUMNS + self.extr.TSIM_COLUMNS + self.extr.WSIM_COLUMNS):
            X[col] = X[col].astype(np.float32)

        for col in self.extr.KEYWORD_COLUMNS:
            X[col] = X[col].astype(np.uint8)

        # Train ETR
        self.regs = {}
        for target_column in self.conv.COLUMNS:
            # reg = ExtraTreesRegressor(**self.etr_params)
            reg = ExtraTreesRegressor()
            reg.fit(X, Y[target_column])
            self.regs[target_column] = reg

    def predict(self, test_dialogue):
        features_list = self.extr.extract(test_dialogue)

        X = []
        for feature_i, features in enumerate(features_list):
            turn = test_dialogue["turns"][feature_i]
            if turn['speaker'] == "U" or turn['annotations'] == []:
                continue
            else:
                X.append(features)

        X = pd.DataFrame(X, columns=self.extr.COLUMNS)
        for col in (self.extr.TSIM_COLUMNS + self.extr.WSIM_COLUMNS):
            X[col].fillna(self.X_mean[col], inplace=True)

        # Convert dtypes
        for col in (self.extr.STANDALONE_COLUMNS + self.extr.TSIM_COLUMNS + self.extr.WSIM_COLUMNS):
            X[col] = X[col].astype(np.float32)

        for col in self.extr.KEYWORD_COLUMNS:
            X[col] = X[col].astype(np.uint8)

        target_values_list = np.array([self.regs[col].predict(X) for col in self.conv.COLUMNS]).T

        Y = []
        for target_values in target_values_list:
            Y.append(self.conv.restore(target_values))

        return Y

    def save(self, save_dir):
        import pickle
        # from sklearn.externals import joblib
        for target, reg in self.regs.items():
            model_path = os.path.join(save_dir, "%s.etr.pkl" % target)
            pickle.dump(reg, open(model_path, 'wb'))
            # joblib.dump(reg, model_path)

    def restore(self, save_dir):
        import pickle
        # from sklearn.externals import joblib
        self.regs = {}
        for target in self.conv.COLUMNS:
            model_path = os.path.join(save_dir, "%s.etr.pkl" % target)
            self.regs[target] = pickle.load(open(model_path, 'rb'))
            # self.regs[target] = joblib.load(model_path)
