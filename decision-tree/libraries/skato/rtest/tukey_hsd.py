# -*- coding: utf-8 -*-

import os
import logging
import math
import numpy as np
import pandas as pd
import pyper


class RTest():

    def __init__(self):
        self.r = pyper.R(use_pandas='True')
        logging.info(self.r("library(reshape2)"))

    def tukey_hsd_test(self, csv_path):
        self.r.assign("jsd", pd.read_csv(csv_path))
        self.r('jsd$Utterance <- rownames(jsd)')
        logging.debug("Melting for %s" % csv_path)
        self.r('jsdmelt <- melt(jsd, id.vars="Utterance", variable.name="Run", value.name="JSD")')
        logging.debug("Calculating ANOVA for %s" % csv_path)
        self.r('jsd_aov_res <- aov(JSD ~ factor(Run) + factor(Utterance), data=jsdmelt)')
        self.r('summary_jsd_aov_res <- summary(jsd_aov_res)')
        logging.debug("Testing TukeyHSD for %s" % csv_path)
        self.r('jsd_hsd_res <- TukeyHSD(jsd_aov_res)')
        self.r('jsd_hsd_factor_run <- jsd_hsd_res$`factor(Run)`')

        logging.debug("Getting summary_jsd_aov_res from R")
        summary_jsd_aov_res = self.r.get("summary_jsd_aov_res")
        logging.debug("Got summary_jsd_aov_res from R")
        logging.debug("Getting jsd_hsd_factor_run from R")
        jsd_hsd_factor_run = self.r.get("jsd_hsd_factor_run")
        self.r('jsd_hsd_factor_run_colnames <- colnames(jsd_hsd_factor_run)')
        jsd_hsd_factor_run_colnames = self.r.get("jsd_hsd_factor_run_colnames")
        self.r('jsd_hsd_factor_run_rownames <- rownames(jsd_hsd_factor_run)')
        jsd_hsd_factor_run_rownames = self.r.get("jsd_hsd_factor_run_rownames")
        logging.debug("Got jsd_hsd_factor_run from R")

        ve = summary_jsd_aov_res[0][" Mean Sq "].iloc[2]
        m = pd.DataFrame(jsd_hsd_factor_run,
                         index=jsd_hsd_factor_run_rownames,
                         columns=jsd_hsd_factor_run_colnames)
        m["es"] = m["diff"] / math.sqrt(ve)

        output_path = csv_path + ".tukeyhsd.csv"
        m.to_csv(output_path)

        return m, summary_jsd_aov_res[0]
