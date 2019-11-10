# -*- coding: utf-8 -*-


import os
import datetime
from collections import defaultdict, Counter
import pandas as pd
from sklearn.model_selection import train_test_split


from .json import Jsons
from .annotation import Annotation


def load_dialogues(data_dir, corpora, max_n=None):
    corpus_dialogues = {}
    for corpus in corpora:
        corpus_dir = os.path.join(data_dir, corpus)
        dialogues = Dialogues(corpus_dir, max_n=max_n)
        for d_i in range(len(dialogues)):
            dialogues[d_i]["corpus"] = corpus
        corpus_dialogues[corpus] = dialogues
    return corpus_dialogues


def devide_dialogues_dict(dialogues, key="corpus"):
    dialogues_dict = defaultdict(list)
    for dialogue in dialogues:
        dialogues_dict[dialogue[key]].append(dialogue)
    return dialogues_dict


def split_dialogues_dict(dialogues_dict, test_size=0.5):
    keys = dialogues_dict.keys()
    train_dialogues_dict = {}
    test_dialogues_dict = {}
    for key in keys:
        train_dialogues, test_dialogues = train_test_split(
            dialogues_dict[key], test_size=test_size, random_state=datetime.datetime.now().microsecond)
        train_dialogues_dict[key] = Dialogues.from_list(train_dialogues)
        test_dialogues_dict[key] = Dialogues.from_list(test_dialogues)

        # import ipdb
        # ipdb.set_trace()

    return train_dialogues_dict, test_dialogues_dict


class Dialogue(dict):

    def __init__(self, dialogue_json):
        dict.__init__(self)
        self.update(dialogue_json)

    def add_tokens(self, tokenizer):
        for turn_i in range(len(self["turns"])):
            tokens = tokenizer.tokenize(self["turns"][turn_i]["utterance"])
            self["turns"][turn_i]["tokens"] = tokens

    def add_features(self, extractors):
        for extractor in extractors:
            if extractor.DOMAIN == "DIALOGUE":
                features_list = extractor.extract(self)
                for turn_i in range(len(self["turns"])):
                    features = features_list[turn_i]
                    for col, feature in zip(extractor.OUTPUT_COLUMNS, features):
                        self["turns"][turn_i][col] = feature
            elif extractor.DOMAIN == "TURN":
                for turn_i in range(len(self["turns"])):
                    features = extractor.extract(self["turns"][turn_i])
                    for col, feature in zip(extractor.OUTPUT_COLUMNS, features):
                        self["turns"][turn_i][col] = feature
            else:
                raise Exception("%s is invalud domain for extractor" % extractor.DOMAIN)

    def to_df(self):
        data = defaultdict(list)

        eval_index = 0
        for turn in self['turns']:

            data["DIALOGUE-ID"].append(self["dialogue-id"])
            data["KEY"].append("%s-%d" % (self["dialogue-id"], turn["turn-index"]))
            data["CORPUS"].append(self["corpus"])

            for key in turn.keys():
                if not isinstance(turn[key], list):
                    data[key.upper()].append(turn[key])

            if turn['speaker'] == "U" or turn['annotations'] == []:  # modified Sep 17 2017
                data["EVAL_INDEX"].append(-1)  # not None

            else:
                data["EVAL_INDEX"].append(eval_index)
                eval_index += 1

        df = pd.DataFrame(data)
        df["EVAL_INDEX"] = df["EVAL_INDEX"].astype(int)
        return df

    def tokenized_turns(self, tokenizer):  # for Old version
        turns = []
        for turn_i, turn in enumerate(self["turns"]):
            if turn_i != turn["turn-index"]:
                raise Exception("turn_i != turn['turn-index']")
            speaker = turn["speaker"]
            terms = tokenizer.tokenize(turn["utterance"])
            turns.append({"speaker": speaker, "terms": terms})
        return turns

    def dist_list(self, all=True):  # for Old version
        """
        label order: ["O", "T", "X"]
        """
        dists = []
        for turn in self["turns"]:
            breakdowns = [a["breakdown"] for a in turn["annotations"]]
            if all:
                if len(breakdowns) > 0:
                    counter = Counter(breakdowns)
                    s = float(sum(counter.values()))
                    dist = [counter.get(l, 0) / s for l in ["O", "T", "X"]]
                    dists.append(dist)
                else:
                    dists.append([])
            else:
                if turn["speaker"] == "S" and len(breakdowns) > 0:
                    counter = Counter(breakdowns)
                    s = float(sum(counter.values()))
                    dist = [counter.get(l, 0) / s for l in ["O", "T", "X"]]
                    dists.append(dist)
        return dists


class Dialogues(Jsons):

    # FILENAME = "*log.json"
    # FILENAME = "*labels.json"
    FILENAME = "*.json"
    CLASS = Dialogue

    @classmethod
    def from_list(self, items):
        dialogues = Dialogues("")
        # import ipdb
        # ipdb.set_trace()
        for item in items:
            dialogues.append(Dialogue(item))
        # import ipdb
        # ipdb.set_trace()
        return dialogues
