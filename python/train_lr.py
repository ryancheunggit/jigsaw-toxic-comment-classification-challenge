#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

# https://www.kaggle.com/yekenot/toxic-regression/code

import pandas as pd
import numpy as np
import argparse
import os
import gc
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix
from constants import *
from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from tqdm import tqdm
tqdm.pandas(desc="progress")


def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()


def main():
    parser = argparse.ArgumentParser(description='training logistic regression on bag of words features and make predictions')
    parser.add_argument('--max_word_features', type = int, default = -1, help = 'max number of words for word vectorizer')
    parser.add_argument('--max_word_ngram', type = int, default = 3, help = 'ngram range for word vectorizer')
    parser.add_argument('--max_char_features', type = int, default = -1, help = 'max number of words for char vectorizer')
    parser.add_argument('--max_char_ngram', type = int, default = 6, help = 'ngram range for char vectorizer')
    parser.add_argument('--C', type = float, default = -1, help = 'C penalty for logistic regression, -1 for grid search')
    parser.add_argument('--save_prediction', type = str2bool, default = "False", help = 'whether to save prediction')
    parser.add_argument('--num_folds', type = int, default = 5, help = "k in kfold")
    parser.add_argument('--kfold_seed', type = int, default = 2014, help = "seed number to split for kfold")
    parser.add_argument('--preprocessing', type = str2bool, default = 'False', help = "use preprocessed data or not")
    args = parser.parse_args()
    print(args)

    if args.max_word_features < 0:
        args.max_word_features = None
    if args.max_char_features < 0:
        args.max_char_features = None

    # print("=== read in data")
    train, test, y, y_label_dist = load_data(processed = args.preprocessing)
    sub = pd.read_csv('../input/sample_submission.csv')

    train['comment_text'].fillna("__UNKNOWN__", inplace = True)
    test['comment_text'].fillna("__UNKNOWN__", inplace = True)
    all_text = pd.concat([train['comment_text'], test['comment_text']], axis=0)

    print("extracting word features {}".format(datetime.now()))
    word_vectorizer = TfidfVectorizer(
            tokenizer = tokenize,
            analyzer = 'word',
            lowercase = 'True',
            min_df = 5,
            max_features = args.max_word_features,
            ngram_range = (1, args.max_word_ngram),
            stop_words='english',
            sublinear_tf = True,
            strip_accents='unicode',
            use_idf = True
        )

    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train['comment_text'])
    test_word_features = word_vectorizer.transform(test['comment_text'])
    print("word feature vocab size : {}".format(len(word_vectorizer.vocabulary_)))

    print("extracting char features {}".format(datetime.now()))
    # print("=== fitting char vectorizer")
    char_vectorizer = TfidfVectorizer(
            analyzer = 'char',
            min_df = 3,
            max_features = args.max_char_features,
            ngram_range = (1, args.max_char_ngram),
            sublinear_tf = True,
            strip_accents = 'unicode'
        )

    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train['comment_text'])
    test_char_features = char_vectorizer.transform(test['comment_text'])
    print("word feature vocab size : {}".format(len(char_vectorizer.vocabulary_)))

    # print("=== concat features")
    train_features = hstack([train_char_features, train_word_features], format='csr')
    test_features = hstack([test_char_features, test_word_features], format='csr')

    # print("=== free up mem")
    del train_word_features, train_char_features, test_word_features, test_char_features
    gc.collect()

    # === cv splits and place holders
    splitter = StratifiedKFold(n_splits = args.num_folds, shuffle = True, random_state = args.kfold_seed)
    folds = list(splitter.split(train_features, y_label_dist))

    if args.C == -1:
        print("doing grid search to tune logistic regression per label")
        classifier_params = {}
        aucs = []
        # per label logistic regression tuning
        for idx, label in enumerate(LABELS):
            print("idx {} label {} started cv at time {}".format(idx, label, datetime.now()))
            lr_model = LogisticRegression(fit_intercept = True, penalty = 'l2')
            param_grid = {
                'C':   [1, 3, 5, 7 ],
                'tol': [0.05, 0.1, 0.5],
                'solver': ['lbfgs', 'newton-cg'],
                'class_weight': ['balanced']
                }
            grid_search = GridSearchCV(
                    estimator = lr_model
                    , param_grid = param_grid
                    , scoring = 'roc_auc'
                    , n_jobs = 3
                    , cv = folds
                    , refit = True
                    , verbose = 1
                    , return_train_score = True
                )
            results = grid_search.fit(train_features, y[:, idx])
            print(results.best_score_)
            classifier_params[label] = results.best_params_
            aucs.append(results.best_score_)
        print(np.mean(aucs), np.std(aucs))

    train_metas = np.zeros(y.shape)
    aucs = []
    losses = []
    test_probs = []
    classifiers = {}
    for fold_num, [train_indices, valid_indices] in enumerate(folds):
        print("=== fitting fold {} datetime {} ===".format(fold_num, datetime.now()))
        x_train, x_valid = train_features[train_indices,:], train_features[valid_indices,:]
        y_train, y_valid = y[train_indices], y[valid_indices]

        valid_preds = np.zeros(y_valid.shape)
        test_preds = np.zeros((test_features.shape[0], len(LABELS)))

        for idx, label in enumerate(LABELS):
            print("fitting logreg for label {} at time {}".format(label, datetime.now()))
            classifier = "fold_{}_{}".format(fold_num, label)
            if args.C != -1:
                classifiers[classifier] = LogisticRegression(
                        solver = 'sag',
                        C = args.C
                    )
            else:
                classifiers[classifier] = LogisticRegression(
                        C = classifier_params[label]['C'],
                        class_weight = classifier_params[label]['class_weight'],
                        solver = classifier_params[label]['solver'],
                        tol = classifier_params[label]['tol']
                    )

            classifiers[classifier].fit(x_train, y_train[:, idx])
            valid_preds[:, idx] = classifiers[classifier].predict_proba(x_valid)[:, 1]
            test_preds[:, idx] = classifiers[classifier].predict_proba(test_features)[:, 1]

        train_metas[valid_indices] = valid_preds
        test_probs.append(test_preds)
        auc_score = roc_auc_score(y_valid, valid_preds)
        log_loss_score = avg_log_loss(y_valid, valid_preds)

        print("validation auc {} log loss {}".format(auc_score, log_loss_score))
        aucs.append(auc_score)
        losses.append(log_loss_score)

    print("mean auc score: {} - std {} , mean log loss score: {} - std {}".format(
            np.mean(aucs), np.std(aucs), np.mean(losses), np.std(losses)
        ))

    if args.save_prediction:
        out_dir = "../models/log_reg-w-{}-{}-c-{}-{}-C-{}-k-{}".format(
                args.max_word_features,
                args.max_word_ngram,
                args.max_char_features,
                args.max_char_ngram,
                args.C,
                args.num_folds
            )
        if args.preprocessing:
            out_dir += '-p'
        try:
            os.mkdir(out_dir)
        except:
            print("path for ouputs already exists")

        pd.DataFrame(train_metas, columns = LABELS).to_csv(out_dir + "/train_meta_probs_round_0.csv", index = False)
        sub[LABELS] = np.zeros(sub[LABELS].shape)
        for i in range(5):
            sub[LABELS] += test_probs[i]
        sub[LABELS] /= 5
        sub.to_csv(out_dir + "/test_probs_5_bag_arith_mean_round_0.csv", index = False)


if __name__ == '__main__':
    main()
