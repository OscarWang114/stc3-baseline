# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


def prepare_XSY(df, extr, conv, corpus_order=None):
    eval_df = df[df["EVAL_INDEX"] >= 0].reset_index(drop=True)
    meta = eval_df[["DIALOGUE-ID", "KEY", "CORPUS", "TURN-INDEX", "SPEAKER", "TIME"]]
    X = eval_df[extr.name + ["CORPUS"]]
    Y = conv.convert(eval_df)

    # Fill NA
    fillna_cols = extr.TSIM_COLUMNS + extr.WSIM_COLUMNS
    X_mean = X[fillna_cols].mean()
    for col in fillna_cols:
        X[col] = X[col].fillna(X_mean[col])

    # Convert dtypes
    float32_cols = [col for ftype in ["STANDALONE", "TSIM", "WSIM"]
                    for col in getattr(extr, "%s_COLUMNS" % ftype)]
    for col in float32_cols:
        X[col] = X[col].astype(np.float32)

    uint_cols = extr.KEYWORD_COLUMNS
    for col in uint_cols:
        X[col] = X[col].astype(np.uint8)

    if corpus_order:
        S = pd.get_dummies(X["CORPUS"])[corpus_order]
    else:
        S = pd.get_dummies(X["CORPUS"])

    return X[extr.name], S, Y, meta
