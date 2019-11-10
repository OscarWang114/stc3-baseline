# -*- coding: utf-8 -*-


import os
import glob
import json
import codecs
import logging
from collections import defaultdict


from .common import Dialogue


def load_train_dialogue_list(**kwargs):
    logging.info("Loading training dataset")
    train_dialogue_list = defaultdict(list)
    for corpus, dialogue in DialogueIterator(kwargs["dev_dir"], lang=kwargs["lang"]):
        dialogue["corpus"] = corpus
        train_dialogue_list[corpus].append(dialogue)
    logging.info("Loaded training dataset")
    return train_dialogue_list


def load_test_dialogue_list(**kwargs):
    logging.info("Loading testing dataset")
    test_dialogue_list = defaultdict(list)
    eval_lang = "eval_%s" % kwargs["lang"]
    for corpus, dialogue in DialogueIterator(kwargs["eval_dir"], lang=eval_lang):
        dialogue["corpus"] = corpus
        test_dialogue_list[corpus].append(dialogue)
    logging.info("Loaded testing dataset")
    return test_dialogue_list


def train_and_test_gen(**kwargs):
    train_dialogue_list = load_train_dialogue_list(**kwargs)
    test_dialogue_list = load_test_dialogue_list(**kwargs)

    for corpus, test_dialogues in test_dialogue_list.items():
        if kwargs["train_mode"] == "samecorpus":
            train_dialogues = train_dialogue_list[corpus]
        elif kwargs["train_mode"] == "allcorpus":
            train_dialogues = [dialogue for dialogues in train_dialogue_list.values()
                               for dialogue in dialogues]
        else:
            raise Exception("%s is unvalid training mode" % kwargs["train_mode"])

        yield corpus, test_dialogues, train_dialogues


class DialogueIterator(object):

    CORPUS_FOLDER_NAME_MAP = {
        "json_data_100": "TickTock",
        "IRIS_json_data": "IRIS",
        "CIC_json_data": "CIC",
        "YI_json_data": "YI",
        "DBDC2_dev/DCM": "DCM",
        "DBDC2_dev/DIT": "DIT",
        "DBDC2_dev/IRS": "IRS",
        "DBDC2_ref/DCM": "DCM",
        "DBDC2_ref/DIT": "DIT",
        "DBDC2_ref/IRS": "IRS",
        "dev/dev": "DCM",
        "eval/eval/eval": "DCM",
        "projectnextnlp-chat-dialogue-corpus/json/init100": "DCM",
        "projectnextnlp-chat-dialogue-corpus/json/rest1046": "DCM",
    }

    def __init__(self, data_dir, lang="en", extra=False):
        if lang == "en":
            self.data_dir = os.path.join(data_dir, lang)
            folders = glob.glob(os.path.join(self.data_dir, "*"))
            corpora = [os.path.basename(f) for f in folders if os.path.isdir(f)]
            self.pathes = {}

            for c in corpora:
                dialogues = [os.path.basename(d) for d in glob.glob(
                    os.path.join(self.data_dir, c, "*.log.json"))]
                self.pathes[c] = dialogues

        elif lang == "jp":
            if data_dir.split("/")[-1] == "dev":

                folders = ["DBDC2_dev/DCM",
                           "DBDC2_dev/DIT",
                           "DBDC2_dev/IRS",
                           "DBDC2_ref/DCM",
                           "DBDC2_ref/DIT",
                           "DBDC2_ref/IRS"]
                if extra:
                    folders.extend(["dev/dev",
                                    "eval/eval/eval",
                                    "projectnextnlp-chat-dialogue-corpus/json/init100",
                                    "projectnextnlp-chat-dialogue-corpus/json/rest1046"])
                self.pathes = {}

                self.data_dir = os.path.join(data_dir, lang)
                for f in folders:
                    dialogues = [os.path.basename(d) for d in glob.glob(
                        os.path.join(self.data_dir, f, "*.log.json"))]
                    self.pathes[f] = dialogues

        else:
            self.data_dir = os.path.join(data_dir, lang)
            folders = glob.glob(os.path.join(self.data_dir, "*"))
            corpora = [os.path.basename(f) for f in folders if os.path.isdir(f)]
            self.pathes = {}

            for c in corpora:
                dialogues = [os.path.basename(d) for d in glob.glob(
                    os.path.join(self.data_dir, c, "*.log.json"))]
                self.pathes[c] = dialogues

        logging.info("Data directory: %s" % self.data_dir)
        for c, d in self.pathes.items():
            logging.info("%d dialogues in %s corpus mapped to %s" %
                         (len(d), c, self.CORPUS_FOLDER_NAME_MAP.get(c, c)))

    def __iter__(self):
        # /home/sosukedb/data/dbdc3/dev/en/YI_json_data/YI0053.log.json
        for corpus in self.pathes.keys():
            for dpath in self.pathes[corpus]:
                dialogue = json.load(codecs.open(
                    os.path.join(self.data_dir, corpus, dpath), "r", "utf-8"))

                yield self.CORPUS_FOLDER_NAME_MAP.get(corpus, corpus), Dialogue(dialogue)
