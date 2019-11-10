# -*- coding: utf-8 -*-


import os
import logging
import glob
import codecs
import json
import subprocess
import math
from collections import defaultdict
import numpy as np
import pandas as pd


def calc_distribution(annotations):
    count_O = 0
    count_T = 0
    count_X = 0

    for annotation in annotations:
        if annotation['breakdown'] == 'O':
            count_O += 1
        elif annotation['breakdown'] == 'T':
            count_T += 1
        elif annotation['breakdown'] == 'X':
            count_X += 1

    prob_O = count_O * 1.0 / (count_O + count_T + count_X)
    prob_T = count_T * 1.0 / (count_O + count_T + count_X)
    prob_X = count_X * 1.0 / (count_O + count_T + count_X)

    return [prob_O, prob_T, prob_X]


def majority_label(prob_O, prob_T, prob_X):

    if prob_O >= prob_T and prob_O >= prob_X:
        return "O"
    elif prob_T >= prob_O and prob_T >= prob_X:
        return "T"
    elif prob_X >= prob_T and prob_X >= prob_O:
        return "X"
    else:
        return "O"


def kld(p, q):
    k = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            k += p[i] * (math.log(p[i] / q[i], 2))

    return k


def jsd(p, q):
    m = []
    for i in range(len(p)):
        m.append((p[i] + q[i]) / 2.0)

    return (kld(p, m) + kld(q, m)) / 2.0


def mse(p, q):
    total = 0.0

    for i in range(len(p)):
        total += pow(p[i] - q[i], 2)

    return total / len(p)


def eval_dialogue_dir(eval_dir, data_dir):
    file_num = 0
    data_files = glob.glob(os.path.join(data_dir, "*log.json"))

    all_result = defaultdict(list)

    for f in data_files:
        file_num += 1

        data_fp = codecs.open(f, "r", "utf-8")
        data_json = json.load(data_fp)
        data_fp.close()

        dlg_id = data_json["dialogue-id"]
        eval_fp = codecs.open(os.path.join(eval_dir, dlg_id + ".labels.json"), "r", "utf-8")
        eval_json = json.load(eval_fp)
        eval_fp.close()

        result = eval_dialogue(eval_json, data_json)

        res_len_list = [len(v) for v in result.values()]
        if not all(np.array(res_len_list) == res_len_list[0]):
            raise Exception("not all(np.array(res_len_list) == res_len_list[0])")

        for k, v in result.items():
            all_result[k].extend(v)

    return all_result


def eval_dialogue(eval_json, data_json):
    result = defaultdict(list)

    eval_index = 0
    for turn in data_json['turns']:
        if turn['speaker'] == "U" or turn['annotations'] == []:  # modified Sep 17 2017
            continue

        ans_prob_dist = calc_distribution(turn['annotations'])
        ans_label = majority_label(
            ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2])

        target_label = eval_json['turns'][eval_index]['labels'][0]

        pred_prob_dist = [float(target_label['prob-O']),
                          float(target_label['prob-T']), float(target_label['prob-X'])]

        jsd_O_T_X = jsd(ans_prob_dist, pred_prob_dist)
        jsd_O_TX = jsd([ans_prob_dist[0], ans_prob_dist[1] + ans_prob_dist[2]],
                       [pred_prob_dist[0], pred_prob_dist[1] + pred_prob_dist[2]])
        jsd_OT_X = jsd([ans_prob_dist[0] + ans_prob_dist[1], ans_prob_dist[2]],
                       [pred_prob_dist[0] + pred_prob_dist[1], pred_prob_dist[2]])

        mse_O_T_X = mse(ans_prob_dist, pred_prob_dist)
        mse_O_TX = mse([ans_prob_dist[0], ans_prob_dist[1] + ans_prob_dist[2]],
                       [pred_prob_dist[0], pred_prob_dist[1] + pred_prob_dist[2]])
        mse_OT_X = mse([ans_prob_dist[0] + ans_prob_dist[1], ans_prob_dist[2]],
                       [pred_prob_dist[0] + pred_prob_dist[1], pred_prob_dist[2]])

        pred_label = target_label['breakdown']

        result["DIALOGUE_ID"].append(data_json["dialogue-id"])
        result["TURN_INDEX"].append(turn["turn-index"])
        result["JSD(O,X,T)"].append(jsd_O_T_X)
        result["JSD(O,XT)"].append(jsd_O_TX)
        result["JSD(OX,T)"].append(jsd_OT_X)
        result["MSE(O,X,T)"].append(mse_O_T_X)
        result["MSE(O,XT)"].append(mse_O_TX)
        result["MSE(OX,T)"].append(mse_OT_X)
        result["GOLD_LABEL"].append(ans_label)
        result["PRED_LABEL"].append(pred_label)

        eval_index += 1

    return result


def extract_condition(result_path):
    cond = [s for s in result_path.split("/")
            if "corpus" in s and "feature" in s and "conv" in s][0]

    values = cond.split("_")
    train_mode = values[0].replace("corpus", "")
    feature_type = values[2]
    conv_mode = values[4]

    return cond, train_mode, feature_type, conv_mode


def parse_eval_output(output):
    data = {}
    metrics = []

    lines = output.decode("utf-8").split("\n")
    for line in lines:

        if ":" in line:
            values = line.split(":")

            metric = values[0].rstrip()

            score = values[1].replace("\t", "").replace("\n", "").split(" ")[1]

            try:

                if "." in score:
                    score = float(score)
                else:
                    score = int(score)
            except Exception as e:
                import ipdb
                ipdb.set_trace()

            data[metric] = score
            metrics.append(metric)

    return metrics, data


def get_result(ref_dir, res_dir, corpora=None):
    if not isinstance(corpora, list):
        raise Exception("not isinstance(corpora, list)")

    eval_script_path = os.path.join(
        os.getenv("HOME"), ".ghq/github.com/sosuke-k/d-thesis/scripts/dbdc3/eval.py")

    score = {}
    metrics = None

    for corpus in corpora:
        corpus_ref_dir = os.path.join(ref_dir, corpus)
        corpus_res_dir = os.path.join(res_dir, corpus)
        cmd = ["python", eval_script_path, "-p", corpus_ref_dir, "-o", corpus_res_dir]
        logging.debug("Running %s" % cmd)
        comp = subprocess.run(cmd, stdout=subprocess.PIPE)
        metrics2, score_dict = parse_eval_output(comp.stdout)

        if metrics:
            if metrics != metrics2:
                raise Exception("%s != %s" % (metrics, metrics2))
        else:
            metrics = metrics2

        score[corpus] = score_dict

    macro_score_dict = {}
    for metric in metrics:
        macro_score_dict[metric] = np.mean([score[corpus][metric] for corpus in corpora])

    result = []
    index = []
    cond, train_mode, feature_type, conv_mode = extract_condition(res_dir)
    for k, v in zip(["mode", "feature", "conv"], [train_mode, feature_type, conv_mode]):
        result.append(v)
        index.append(k)
    for metric in metrics:
        result.append(macro_score_dict[metric])
        index.append("%s <MACRO AVERAGE>" % metric)
        for corpus in corpora:
            result.append(score[corpus][metric])
            index.append("%s <%s>" % (metric, corpus))

    res_name = res_dir.split("/")[-2]

    df = pd.DataFrame({res_name: result}, index=index)
    import ipdb
    ipdb.set_trace()

    return pd.DataFrame({res_name: result}, index=index)
