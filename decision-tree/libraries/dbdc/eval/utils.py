# -*- coding: utf-8 -*-


import copy
from collections import defaultdict
import numpy as np
import pandas as pd


from dbdc.distribution.distances import mse, jsd, nmd, rsnod


def jsd_list(gold_dialogue, pred_dialogue):
    _jsd_list = []

    for turn_i, turn in enumerate(gold_dialogue["turns"]):
        if turn['speaker'] == "U" or turn['annotations'] == []:
            continue
        else:
            gold_prob = gold_dialogue["turns"][turn_i]["prob"]
            pred_prob = pred_dialogue["turns"][turn_i]["prob"]
            _jsd_list.append(jsd(gold_prob, pred_prob))

    return _jsd_list


def majority_label(prob_O, prob_T, prob_X, threshold=0.0):

    if prob_O >= prob_T and prob_O >= prob_X and prob_O >= threshold:
        return "O"
    elif prob_T >= prob_O and prob_T >= prob_X and prob_T >= threshold:
        return "T"
    elif prob_X >= prob_T and prob_X >= prob_O and prob_X >= threshold:
        return "X"
    else:
        return "O"


def eval_dist(pred_Y, gold_Y,
              metrics=["JSD", "MSE", "NMD", "RSNOD"],
              dist_columns=["PROB-%s" % l for l in ["O", "T", "X"]],
              pred_prefix="PRED_", gold_prefix="GOLD_"):

    def check_columns_exist(columns, df):
        all_exist = True
        df_columns = list(df.columns)
        for col in columns:
            if not col in df_columns:
                all_exist = False
        return all_exist

    pred_colmap = {col: pred_prefix + col for col in dist_columns}
    if not check_columns_exist(list(pred_colmap.values()), pred_Y):
        Y = pred_Y.rename(columns=pred_colmap)
    else:
        Y = pred_Y

    gold_colmap = {col: gold_prefix + col for col in dist_columns}
    if not check_columns_exist(list(gold_colmap.values()), gold_Y):
        G = gold_Y.rename(columns=gold_colmap)
    else:
        G = gold_Y

    probs = Y.join(G)

    metric_funcs = {
        "JSD": jsd,
        "MSE": mse,
        "NMD": nmd,
        "RSNOD": rsnod,
    }

    def eval_func(row):
        pred_dist = [row[pred_colmap[col]]
                     for col in dist_columns]
        pred_majority_label = majority_label(*pred_dist)
        gold_dist = [row[gold_colmap[col]]
                     for col in dist_columns]
        gold_majority_label = majority_label(*gold_dist)
        values_to_add = [pred_majority_label, gold_majority_label]
        values_to_add.extend([metric_funcs[met](pred_dist, gold_dist)
                              for met in metrics])
        return tuple(values_to_add)

    results = probs.apply(eval_func, axis=1, result_type="expand")

    columns_to_add = ["PRED", "GOLD"] + metrics
    results.rename(columns={i: col for i, col in enumerate(columns_to_add)},
                   inplace=True)

    return results


def label_eval_score(df):
    res = defaultdict(list)

    for corpus in set(df["CORPUS"]):
        # df["CURRENT_CORPUS"] = corpus
        results = df[np.array(df["CORPUS"] == corpus)]
        # results["BREAKDOWN_LABEL"] = "X"
        predX_ansX = len(results[np.array(results["PRED"] == "X")
                                 & np.array(results["GOLD"] == "X")])
        predX = len(results[np.array(results["PRED"] == "X")])
        ansX = len(results[np.array(results["GOLD"] == "X")])

        precision = 0.0
        recall = 0.0
        fmeasure = 0.0
        if predX_ansX > 0:
            if predX > 0:
                precision = predX_ansX * 1.0 / predX
            if ansX > 0:
                recall = predX_ansX * 1.0 / ansX

        if precision > 0 and recall > 0:
            fmeasure = (2 * precision * recall) / (precision + recall)

        correct_num = len(results[results["PRED"] == results["GOLD"]])
        acc = correct_num * 1.0 / len(results)

        res["CORPUS"].append(corpus)
        res["ACC"].append(acc)
        res["PRE"].append(precision)
        res["REC"].append(recall)
        res["F"].append(fmeasure)

    res = pd.DataFrame(res)
    res = res.set_index("CORPUS")
    return res


# def show_results(model, conv, X, Y, S):
#     results = eval_dist(conv.restore(model.predict(X)),
#                         copy.deepcopy(Y))
# def show_results(model, conv, X, Y, S):
def show_results(results):
    """
    results columns must include "PRED", "GOLD" and "CORPUS"
    """
    # results = eval_dist(conv.restore(model.predict(X)),
    #                     copy.deepcopy(Y))
    # results = results.join(S[["CORPUS"]])

    def show_corpus_scores(corpus_scores):
        print(corpus_scores)
        print(corpus_scores.describe().loc[["mean"]].reset_index(
            drop=True).rename(index={0: "AVERAGE"}))

    label_score = label_eval_score(results)
    show_corpus_scores(label_score)

    dist_score = results.groupby("CORPUS").mean()
    show_corpus_scores(dist_score)

    return results
