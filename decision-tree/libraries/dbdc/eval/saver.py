# -*- coding: utf-8 -*-


import os
import logging
import codecs
import json
import numpy as np


from dbdc.base.label import BREAKDOWN


from .utils import majority_label


class LabelSaver(object):

    LABELED_JSON_FORMAT = "{did}.labels.json"

    def extract(self, labeled_dialogue):
        labeled_json = {"dialogue-id": labeled_dialogue["dialogue-id"], "turns": []}

        for turn in labeled_dialogue["turns"]:
            if turn['speaker'] == "U" or turn['annotations'] == []:
                continue
            else:
                prob = turn["prob"]
                _label = {"prob-%s" % l: p for l, p in zip(BREAKDOWN.LABELS, prob)}
                _label["breakdown"] = BREAKDOWN.LABELS[np.argmax(prob)]
                labeled_json["turns"].append({
                    "turn-index": turn["turn-index"],
                    "labels": [_label],
                })

        return labeled_json

    def save(self, output_dir, labeled_dialogue):
        dialogue_id = labeled_dialogue["dialogue-id"]
        labeled_json = self.extract(labeled_dialogue)
        output_filename = self.LABELED_JSON_FORMAT.format(did=dialogue_id)
        output_path = os.path.join(output_dir, output_filename)
        logging.debug("Saving to %s" % output_path)
        json.dump(labeled_json, codecs.open(output_path, "w", "utf-8"), indent=2)
        logging.debug("Saved to %s" % output_path)

    def save_results(self, results, save_dir):
        corpora = set(results["CORPUS"])
        for corpus in corpora:
            corpus_dir = os.path.join(save_dir, corpus)
            if not os.path.isdir(corpus_dir):
                os.mkdir(corpus_dir)

            sorted_results = results[results["CORPUS"] == corpus].sort_values(["DIALOGUE-ID", "TURN-INDEX"])
            labeled = {}
            for i in range(len(sorted_results)):
                row = sorted_results.iloc[i]
                dialogue_id = row["DIALOGUE-ID"]
                if not dialogue_id in labeled:
                    labeled[dialogue_id] = {"dialogue-id": dialogue_id, "turns": []}

                dist = [row["PROB-%s" % l] for l in ["O", "T", "X"]]
                breakdown = majority_label(*dist)
                _label = {"prob-%s" % l: p for l, p in zip(["O", "T", "X"], dist)}
                _label["breakdown"] = breakdown
                labeled[dialogue_id]["turns"].append({
                    "turn-index": int(row["TURN-INDEX"]),
                    "labels": [_label],
                })
            for dialogue_id, labeled_json in labeled.items():
                output_filename = self.LABELED_JSON_FORMAT.format(did=dialogue_id)
                output_path = os.path.join(corpus_dir, output_filename)
                logging.debug("Saving to %s" % output_path)
                json.dump(labeled_json, codecs.open(output_path, "w", "utf-8"), indent=2)
                logging.debug("Saved to %s" % output_path)
