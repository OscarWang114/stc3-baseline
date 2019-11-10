# -*- coding: utf-8 -*-


import logging
import codecs
import pickle
import json
from pathlib2 import Path


class AbstractExtractor():

    DOMAIN = ""
    NEED_COLUMNS = []
    OUTPUT_COLUMNS = []

    def __init__(self, **kwargs):
        pass

    def initialize(self):
        logging.warn("%s#initialize function not implemented" %
                     self.__class__.__name__)

    def fit(self, *args, **kwargs):
        logging.warn("%s#fit function not implemented" %
                     self.__class__.__name__)

    def check_columns(self, df):
        input_columns = list(df.columns)
        for col in self.NEED_COLUMNS:
            if not col in input_columns:
                raise Exception("%s not included in %s" % (col, input_columns))
        return True

    def save(self, save_dir):
        logging.warn("%s#save function not implemented" %
                     self.__class__.__name__)

    def restore(self, save_dir):
        logging.warn("%s#restore function not implemented" %
                     self.__class__.__name__)

    def save_to_json(self, output_path, obj):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(obj, codecs.open(output_path, "w", "utf-8"), indent=2)

    def load_from_json(self, output_path):
        return json.load(codecs.open(output_path, "r", "utf-8"))

    def save_to_pkl(self, output_path, obj):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(obj, open(output_path, "wb"))

    def load_from_pkl(self, output_path):
        return pickle.load(open(output_path, "rb"))
