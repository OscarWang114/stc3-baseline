# -*- coding: utf-8 -*-


import os
import logging
import copy
# import json
# import codecs
# from collections import defaultdict
# import numpy as np
import pandas as pd
# from scipy import stats
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")


# from .verifier import Estimator, Params


from dbdc.base.dialogue import devide_dialogues_dict
from dbdc.base.label import BREAKDOWN
from dbdc.base.annotation import Annotation
from dbdc.dataset.corpus import Corpus

# Import Old version
from dbdc.old.predicter import ETRBreakdownPredictor


# class AbstractBreakdownPredictor(object):
#
#     def fit(self):
#         raise Exception("fit function is not implemented")
#
#     def predict(self):
#         raise Exception("predict function is not implemented")
#
#     # def score(self, test_dialogues):
#     #     scores = []
#     #     for test_dialogue in test_dialogues:
#     #         dialogue_id = test_dialogue["dialogue-id"]
#     #         labeled_dialogue = self.predict(test_dialogue)
#     #         import ipdb
#     #         ipdb.set_trace()
#     #         score = Dialogue.score(labeled_dialogue, test_dialogue)
#     #         scores.append(score)
#     #     mean_score = np.mean(scores)
#     #     return mean_score


# class ETRBreakdownPredictor(AbstractBreakdownPredictor):
#
#     pass


class DisvarDataSelectionPrerictor():

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

    def __init__(self, threashold=0.02, extractor=None, converter=None, **kwargs):
        # self.fnames = feature_names
        # self.conv = converter
        self.threashold = threashold
        self.etr_params = {k: kwargs.get(k, self.DEFAULT_ETR_PARAMS[k])
                           for k in self.DEFAULT_ETR_PARAMS.keys()}

        self.extractor = extractor
        self.converter = converter

        self.predictors = {corpus: ETRBreakdownPredictor(extractor, converter, **kwargs)
                           for corpus in Corpus.list(lang=kwargs["lang"])}
        self.global_predictor = ETRBreakdownPredictor(extractor, converter, **kwargs)

        self.type = "global"

    def fit(self, dev_corpus_dialogues):
        # 辞書型で渡す
        # corpus_test_dialogues = devide_dialogues_dict(test_dialogues, key="corpus")
        dfs = []
        for corpus in dev_corpus_dialogues.keys():
            dfs.append(dev_corpus_dialogues[corpus].to_df())
        df = pd.concat(dfs)
        eval_turns = df[df.EVAL_INDEX >= 0]
        dialogues = eval_turns[["DIALOGUE-ID", "CORPUS"]].drop_duplicates(
            subset=["DIALOGUE-ID"], keep="first")

        # EVAL_INDEX >= 0 を対象
        for speaker in ["SYSTEM"]:
            turn_length_group = eval_turns[["DIALOGUE-ID", "TURN_LENGTH"]].groupby("DIALOGUE-ID")
            for stat_name in ["var", "mean", "max", "min"]:
                turn_length_group_stat = getattr(turn_length_group, stat_name)().reset_index()
                column_map = {"TURN_LENGTH": "TURN_LENGTH_%s" % stat_name.upper()}
                turn_length_group_stat.rename(columns=column_map, inplace=True)
                dialogues = pd.merge(dialogues, turn_length_group_stat,
                                     on="DIALOGUE-ID", how="left")

        for dis_col in ["JSD_FROM_DIALOGUE_MEAN"]:
            dialogues_distance_group = eval_turns[["DIALOGUE-ID", dis_col]].groupby("DIALOGUE-ID")
            for stat_name in ["var", "mean", "max", "min"]:
                dialogues_distance_group_stat = getattr(
                    dialogues_distance_group, stat_name)().reset_index()
                column_map = {dis_col: "%s_%s" % (dis_col, stat_name.upper())}
                dialogues_distance_group_stat.rename(columns=column_map, inplace=True)
                dialogues = pd.merge(dialogues, dialogues_distance_group_stat,
                                     on="DIALOGUE-ID", how="left")

        corpus_mean = dialogues.groupby("CORPUS").mean().reset_index()
        X = [[x] for x in corpus_mean["TURN_LENGTH_MIN"]]
        y = corpus_mean["JSD_FROM_DIALOGUE_MEAN_MEAN"]

        self.reg = LinearRegression()
        self.reg.fit(X, y)
        self.corpus_mean = corpus_mean

        # Fit ETR
        # X = df[self.fnames]
        # Y = self.conv.convert_from_df(df["PROB-%s" % l for l in BREAKDOWN.LABELS])
        train_dialogues = [d for dialogues in dev_corpus_dialogues.values() for d in dialogues]
        self.global_predictor.fit(train_dialogues)
        for corpus in dev_corpus_dialogues.keys():
            self.predictors[corpus].fit(dev_corpus_dialogues[corpus])

    def second_fit(self, dialogues):
        df = dialogues.to_df()
        # import ipdb
        # ipdb.set_trace()
        eval_turns = df[df.EVAL_INDEX >= 0]
        dialogues = eval_turns[["DIALOGUE-ID", "CORPUS"]].drop_duplicates(
            subset=["DIALOGUE-ID"], keep="first")

        # EVAL_INDEX >= 0 を対象
        for speaker in ["SYSTEM"]:
            turn_length_group = eval_turns[["DIALOGUE-ID", "TURN_LENGTH"]].groupby("DIALOGUE-ID")
            for stat_name in ["var", "mean", "max", "min"]:
                turn_length_group_stat = getattr(turn_length_group, stat_name)().reset_index()
                column_map = {"TURN_LENGTH": "TURN_LENGTH_%s" % stat_name.upper()}
                turn_length_group_stat.rename(columns=column_map, inplace=True)
                dialogues = pd.merge(dialogues, turn_length_group_stat,
                                     on="DIALOGUE-ID", how="left")

        mean = dialogues.mean()
        # import ipdb
        # ipdb.set_trace()
        X = [[mean["TURN_LENGTH_MIN"]]]
        pred_y = self.reg.predict(X)
        if pred_y[0] >= self.threashold:
            self.type = "global"
        else:
            self.type = "same"

    def predict(self, test_dialogue):
        if self.type == "global":
            predicted_dist_list = self.global_predictor.predict(test_dialogue)
        elif self.type == "same":
            corpus = test_dialogue["corpus"]
            predicted_dist_list = self.predictors[corpus].predict(test_dialogue)
        else:
            raise Exception("")

        predicted_dialogue = copy.deepcopy(test_dialogue)
        dist_i = 0
        for turn_i, turn in enumerate(predicted_dialogue["turns"]):
            if turn['speaker'] == "U" or turn['annotations'] == []:
                predicted_dialogue["turns"][turn_i]["prob"] = []
            else:
                predicted_dialogue["turns"][turn_i]["prob"] = predicted_dist_list[dist_i]
                dist_i += 1

        if dist_i != len(predicted_dist_list):
            raise Exception("dist_i != len(predicted_dist_list)")

        # import ipdb
        # ipdb.set_trace()

        return predicted_dialogue

    def save_regplot(self, **kwargs):
        x_col = "TURN_LENGTH_MIN"
        y_col = "JSD_FROM_DIALOGUE_MEAN_MEAN"
        X = self.corpus_mean[x_col]
        Y = self.corpus_mean[y_col]

        output_name = "regplot_%s_and_%s.png" % (x_col, y_col)
        output_path = os.path.join(kwargs["output_dir"], output_name)

        logging.info("Saving to %s" % output_path)

        # Config
        if "figsize" in kwargs:
            plt.figure(figsize=kwargs["figsize"])
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"][0], kwargs["xlim"][1])
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"][0], kwargs["ylim"][1])

        # colors = sns.hls_palette(4)
        ax = sns.regplot(x=x_col, y=y_col, data=self.corpus_mean)
        for x, y, corpus in zip(X, Y, self.corpus_mean["CORPUS"]):
            plt.text(x, y, corpus)
        plt.savefig(output_path)
        plt.clf()
        logging.info("Saved to %s" % output_path)
