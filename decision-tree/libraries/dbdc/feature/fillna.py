# -*- coding: utf-8 -*-

import os
import logging
import codecs
import pandas as pd


class NAFiller():

    def fit(self, X):
        self.X_columns = []

        naflags = X.isna().any(axis=0)
        if naflags.any() == False:
            logging.debug("All columns do not have NA")
            # return

        # X_mean = X.mean()
        # try:
        #     self.X_mean = X.mean()[naflags]
        # except Exception as e:
        #     logging.info("All not-str columns do not have NA")
        #     return

        self.X_mean = X.mean()

        self.X_columns = list(self.X_mean.index)

    def apply(self, X):
        naflags = X.isna().any(axis=0)
        if naflags.any() == False:
            logging.debug("All columns do not have NA")
            return X

        nacolumns = list(naflags[naflags].index)
        for col in nacolumns:
            if not col in self.X_columns:
                raise Exception("not %s in %s" % (col, self.X_columns))

        for col in nacolumns:
            X[col].fillna(self.X_mean[col], inplace=True)

        return X

    def output_path(self, save_dir):
        return os.path.join(save_dir, "%s.pkl" % self.__class__.__name__)

    def save(self, save_dir):
        output_path = self.output_path(save_dir)
        self.X_mean.to_pickle(output_path)

    def restore(self, save_dir):
        output_path = self.output_path(save_dir)
        self.X_mean = pd.read_pickle(output_path)
        self.X_columns = list(self.X_mean.index)
