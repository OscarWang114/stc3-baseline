# -*- coding: utf-8 -*-


import os
import logging
import pickle
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pathlib2 import Path


class RFManager(dict):

    DEFAULT_RF_PARAMS = {
        # Default 10. Will change to 100 in version 0.22
        "n_estimators": 50,
        "criterion": "mse",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "auto",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        # "min_impurity_split": None, deprecated since scikit-learn version 0.19
        # TODO: Think about bootstrapping
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": 1,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
    }

    def __init__(self, target_names, **kwargs):
        dict.__init__(self)
        self.target_names = target_names
        # Usage: kwargs.get('val', 'default_val')
        self.rf_params = {k: kwargs.get(k, self.DEFAULT_RF_PARAMS[k])
                          for k in self.DEFAULT_RF_PARAMS.keys()}
        self.init()

    def init(self):
        for target_name in self.target_names:
            self[target_name] = RandomForestRegressor(**self.rf_params)

    def train(self, X, Y):
        for target_name in self.target_names:
            if not target_name in Y.columns:
                raise Exception("%s not exists in %s" %
                                (target_name, Y.columns))

        for target_name in self.target_names:
            self[target_name].fit(X, Y[target_name])

    def predict(self, X):
        Y = {}
        for target_name in self.target_names:
            Y[target_name] = self[target_name].predict(X)

        return pd.DataFrame(Y)

    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for target_name, reg in self.items():
            model_path = os.path.join(save_dir, "%s.rf.pkl" % target_name)
            pickle.dump(reg, open(model_path, 'wb'))
            # joblib.dump(reg, model_path)

    def restore(self, save_dir):
        for target_name in self.target_names:
            model_path = os.path.join(save_dir, "%s.rf.pkl" % target_name)
            self[target_name] = pickle.load(open(model_path, "rb"))
            # self[target_name] = joblib.load(model_path)


class SimpleRFPredictor():

    def __init__(self, pre, conv):
        self.pre = pre
        self.conv = conv
        self.rf = RFManager(self.conv.TARGET_COLUMNS)

    def restore(self, save_dir):
        self.rf.restore(save_dir)

    def predict(self, dialogue):
        X, keys = self.pre.apply(dialogue)
        Y = self.conv.restore(self.rf.predict(X))
        return keys.join(Y)
