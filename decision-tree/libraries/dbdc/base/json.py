# -*- coding: utf-8 -*-


import os
import codecs
import glob
import json
import pandas as pd


class Jsons(list):

    FILENAME = "*"

    CLASS = None

    def __init__(self, json_dir, max_n=None):
        list.__init__(self)

        self.dir = json_dir

        self.files = glob.glob(os.path.join(self.dir, self.FILENAME))

        if len(self.files) == 0:
            # raise Exception("No file in %s" % self.dir)
            pass
        else:
            if max_n:
                self.files = self.files[:max_n]

            for f in self.files:
                data_fp = codecs.open(f, "r", "utf-8")
                data_json = json.load(data_fp)
                data_fp.close()

                # TODO:
                # if data_json["dialogue-id"] == "iris_00072":
                #     continue

                self.append(self.CLASS(data_json) if self.CLASS else data_json)

    def to_df(self):
        # import ipdb
        # ipdb.set_trace()
        if len(self) > 0:
            return pd.concat([j.to_df() for j in self])
        else:
            return None
