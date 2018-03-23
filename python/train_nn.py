#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import pandas as pd
import numpy as np
import os
import argparse
from .models import *
from .constants import *
from .util import *
import models

from datetime import datetime
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
tqdm.pandas(desc="progress")

def main():
    parser = argparse.ArgumentParser(description='training nn and make predictions')
    parser.add_argument('--embedding_type', type = str, default = "fastText", help = 'type of embedding to use')
    parser.add_argument('--model_arch', type = str, default = "1dcnn", help = 'nn architecture')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--save_prediction', type = str2bool, default = "True", help = 'whether to save prediction')
    parser.add_argument('--max_length', type = int, default = 500, help = 'max number of words retain in a comment')
    parser.add_argument('--tokenizer', type = str, default = 'treebank_punkt', help = 'the tokenizer to use')
    parser.add_argument('--embedding_trainable', type = str2bool, default = "False", help = "whether the embedding can be fine tuned")
    parser.add_argument('--num_rounds', type = int, default = 1, help = 'number of rounds to do cv for a single model')
    parser.add_argument('--lr_decay_rounds', type = int, default = 3, help = 'number of rounds to do learning rate decay')
    parser.add_argument('--lr_decay_ratio', type = float, default = 0.1, help = 'reduce lr to what amount')
    parser.add_argument('--early_stopping_rounds', type = int, default = 7, help = 'number of rounds to do early stopping')
    parser.add_argument('--early_stopping_metric', type = str, default = 'auc', help = 'metric to watch for early stopping')
    parser.add_argument('--augument', type = str2bool, default = "False", help = "whether to augument training data")
    parser.add_argument('--dropout', type = float, default = 0.2, help = "the amount of dropout")
    parser.add_argument('--num_folds', type = int, default = 5, help = "k in kfold")
    parser.add_argument('--kfold_seed', type = int, default = 2014, help = "seed number to split for kfold")
    parser.add_argument('--fine_tuning', type = str2bool, default = 'False', help = "whether to fine tune with embedding trainable after converge")
    parser.add_argument('--embedding_dropout', type = str2bool, default = 'False', help = "whether to add dropout after embedding layer")
    parser.add_argument('--num_recurrent_units', type = int, default = 128, help = "number of recurrent units")
    parser.add_argument('--num_cnn_filters', type = int, default = 64, help = "number of cnn filters")
    parser.add_argument('--preprocessing', type = str2bool, default = 'False', help = "use preprocessed data or not")
    args = parser.parse_args()
    print(args)

    type_of_embeddings = {
        'fastText': 'crawl-300d-2M.vec',
        'glove': 'glove.840B.300d.txt',
        'word2vec': 'GoogleNews-vectors-negative300.bin',
        'random': None
    }

    embedding_file = type_of_embeddings[args.embedding_type]
    # === read in data
    train, test, y, y_label_dist = load_data(processed = args.preprocessing)
    sub = pd.read_csv("../input/sample_submission.csv")

    # === process data, create word <-> index mapping, map sentences to sequences of word indicies
    print("process train data")
    processed_sentences_train, word_idx_mapping = process_sentences(
            sentences = train["comment_text"],
            word_idx_mapping = {},
            tokenizer = args.tokenizer
        )
    print("process test data")
    processed_sentences_test, word_idx_mapping = process_sentences(
            sentences = test["comment_text"],
            word_idx_mapping = word_idx_mapping,
            tokenizer = args.tokenizer
        )

    print("== size of vocab is {}".format(len(word_idx_mapping)))

    if args.augument:
        print("read in augument text")
        train_de = pd.read_csv("../input/ext/train_de.csv")
        train_es = pd.read_csv("../input/ext/train_es.csv")
        train_fr = pd.read_csv("../input/ext/train_fr.csv")

        print("process de train data")
        processed_de_train, word_idx_mapping = process_sentences(
                sentences = train_de["comment_text"],
                word_idx_mapping = word_idx_mapping,
                tokenizer = args.tokenizer
            )

        print("process es train data")
        processed_es_train, word_idx_mapping = process_sentences(
                sentences = train_es["comment_text"],
                word_idx_mapping = word_idx_mapping,
                tokenizer = args.tokenizer
            )

        print("process fr train data")
        processed_fr_train, word_idx_mapping = process_sentences(
                sentences = train_fr["comment_text"],
                word_idx_mapping = word_idx_mapping,
                tokenizer = args.tokenizer
            )
        print("== size of vocab is {}".format(len(word_idx_mapping.keys())))

    idx_word_mapping = {idx: word for word, idx in list(word_idx_mapping.items())}

    # === load/initialize word embeddings
    X_train, X_test = None, None
    if args.embedding_type == 'random':
        print("embedding not specified, pad sequences to be of same length")
        embedding_word_idx_mapping = None
        embedding_vectors = None
        embedding_matrix = None
        embedding_idx_word_mapping = None
        max_features = len(idx_word_mapping)
        X_train = pad_sequences(processed_sentences_train, maxlen = args.max_length)
        X_test = pad_sequences(processed_sentences_test, maxlen = args.max_length)
        if args.augument:
            X_train_de = pad_sequences(processed_de_train, maxlen = args.max_length)
            X_train_es = pad_sequences(processed_es_train, maxlen = args.max_length)
            X_train_fr = pad_sequences(processed_fr_train, maxlen = args.max_length)
    else:
        print("load embedding")
        embedding_word_idx_mapping, embedding_vectors = load_embeddings(
                word_idx_mapping, embedding_file = embedding_file
            )
        embedding_size = len(embedding_vectors[0])
        embedding_matrix = np.array(embedding_vectors)
        embedding_idx_word_mapping = {idx: word
                for word, idx in list(embedding_word_idx_mapping.items())
            }
        max_features = len(embedding_idx_word_mapping)
        X_train = word_indices_to_embedding_indices(
                processed_sentences_train, idx_word_mapping,
                embedding_word_idx_mapping, args.max_length
            )
        X_test = word_indices_to_embedding_indices(
                processed_sentences_test, idx_word_mapping,
                embedding_word_idx_mapping, args.max_length
            )
        if args.augument:
            X_train_de = word_indices_to_embedding_indices(
                    processed_de_train, idx_word_mapping,
                    embedding_word_idx_mapping, args.max_length
                )
            X_train_es = word_indices_to_embedding_indices(
                    processed_es_train, idx_word_mapping,
                    embedding_word_idx_mapping, args.max_length
                )
            X_train_fr = word_indices_to_embedding_indices(
                    processed_fr_train, idx_word_mapping,
                    embedding_word_idx_mapping, args.max_length
                )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print("X_train is of shape: {}".format(X_train.shape))
    print("X_test is of shape: {}".format(X_test.shape))

    if args.augument:
        X_train_de = np.array(X_train_de)
        X_train_es = np.array(X_train_es)
        X_train_fr = np.array(X_train_fr)

    model_builder = getattr(models, "build_{}_model".format(args.model_arch))

    # === cv splits and place holders
    splitter = StratifiedKFold(n_splits = args.num_folds, shuffle = True, random_state = args.kfold_seed)
    folds = list(splitter.split(X_train, y_label_dist))

    if args.save_prediction:
        out_dir = "../models/e-{}-m-{}-o-{}-l-{}-t-{}-a-{}-k-{}-d-{}-ed-{}".format(
                args.embedding_type,
                args.model_arch,
                args.early_stopping_metric,
                args.max_length,
                args.tokenizer,
                args.augument,
                args.num_folds,
                args.dropout,
                args.embedding_dropout
            )
        if args.preprocessing:
            out_dir += '-p'
        try:
            os.mkdir(out_dir)
        except:
            print("path for ouputs already exists")

    all_aucs = []
    train_metas = None
    test_metas = None
    for num_round in range(args.num_rounds):
        print("===+++=== round {} ===+++===".format(num_round))
        # fit on each fold with early stopping and make oof & test predictions
        train_meta = np.zeros(y.shape)
        best_aucs = []
        best_epochs = []
        test_probs = []
        for fold_num, [train_indices, valid_indices] in enumerate(folds):
            print("=== fitting round {} fold {} datetime {} ===".format(num_round, fold_num, datetime.now()))
            x_train, x_valid = X_train[train_indices], X_train[valid_indices]
            y_train, y_valid = y[train_indices], y[valid_indices]
            if args.augument:
                augument_indices = y_train.sum(axis = 1) > 0
                x_train_de = X_train_de[train_indices][augument_indices]
                x_train_es = X_train_es[train_indices][augument_indices]
                x_train_fr = X_train_fr[train_indices][augument_indices]
                x_train_aug = np.vstack([x_train, x_train_de, x_train_es, x_train_fr])
                y_train_aug = np.vstack([y_train, y_train[augument_indices],
                                     y_train[augument_indices],
                                     y_train[augument_indices]])

            model = model_builder(
                    embedding_matrix,
                    max_features = max_features,
                    embedding_trainable = args.embedding_trainable,
                    max_sentence_length = args.max_length,
                    dropout_rate = args.dropout,
                    embedding_dropout = args.embedding_dropout,
                    num_recurrent_units = args.num_recurrent_units,
                    num_cnn_filters = args.num_cnn_filters
                )
            # model = build_crnn_model(embedding_matrix, max_features = max_features)
            # model = build_textcnn_model(embedding_matrix, max_features = max_features)
            if num_round == 0 and fold_num == 0:
                print(model.summary())
            print("=== fitting model")

            if not args.augument:
                fitted_model, best_auc, best_epoch = fit_and_eval(
                        model,
                        x_train, y_train,
                        x_valid, y_valid,
                        batch_size = args.batch_size,
                        learning_rate_reducer_rounds = args.lr_decay_rounds,
                        lr_decay_ratio = args.lr_decay_ratio,
                        early_stopping_rounds = args.early_stopping_rounds,
                        early_stopping_metric = args.early_stopping_metric
                    )
            else:
                fitted_model, best_auc, best_epoch = fit_and_eval(
                        model,
                        x_train_aug, y_train_aug,
                        x_valid, y_valid,
                        batch_size = args.batch_size,
                        learning_rate_reducer_rounds = args.lr_decay_rounds,
                        lr_decay_ratio = args.lr_decay_ratio,
                        early_stopping_rounds = args.early_stopping_rounds,
                        early_stopping_metric = args.early_stopping_metric
                    )
            if args.fine_tuning:
                fitted_model.save_weights('temp_weights.h5')
                lr = K.eval(fitted_model.optimizer.lr)
                new_model = model_builder(
                        embedding_matrix,
                        max_features = max_features,
                        embedding_trainable = True,
                        max_sentence_length = args.max_length,
                        dropout_rate = args.dropout
                    )
                new_model.load_weights('temp_weights.h5')
                K.set_value(new_model.optimizer.lr, lr / 10)

                print("=== fine tuning model")

                if not args.augument:
                    fitted_model, best_auc, best_epoch = fit_and_eval(
                            new_model,
                            x_train, y_train,
                            x_valid, y_valid,
                            batch_size = args.batch_size,
                            learning_rate_reducer_rounds = args.lr_decay_rounds,
                            lr_decay_ratio = args.lr_decay_ratio,
                            early_stopping_rounds = args.early_stopping_rounds,
                            early_stopping_metric = args.early_stopping_metric
                        )
                else:
                    fitted_model, best_auc, best_epoch = fit_and_eval(
                            new_model,
                            x_train_aug, y_train_aug,
                            x_valid, y_valid,
                            batch_size = args.batch_size,
                            learning_rate_reducer_rounds = args.lr_decay_rounds,
                            lr_decay_ratio = args.lr_decay_ratio,
                            early_stopping_rounds = args.early_stopping_rounds,
                            early_stopping_metric = args.early_stopping_metric
                        )

            best_aucs.append(best_auc)
            best_epochs.append(best_epoch)

            print("=== generating oof")
            train_meta[valid_indices] = model.predict(x_valid, batch_size = args.batch_size, verbose = 1)

            print("=== making test predictions")
            test_probs.append(model.predict(X_test, batch_size = args.batch_size, verbose = 1))

            del model
            K.clear_session()

        print("{} fold cv mean {} and std {}".format(args.num_folds, np.mean(best_aucs), np.std(best_aucs)))
        print(np.mean(best_epochs))
        print("training meta auc score is : {}".format(roc_auc_score(y, train_meta)))

        if args.save_prediction:
            # save oof meta
            pd.DataFrame(train_meta, columns = LABELS).to_csv(out_dir + "/train_meta_probs_round_{}.csv".format(num_round))

            # average 5 folds predictions
            sub[LABELS] = np.zeros(sub[LABELS].shape)
            for i in range(args.num_folds):
                sub[LABELS] += test_probs[i]
            sub[LABELS] /= args.num_folds
            sub.to_csv(out_dir + "/test_probs_{}_bag_arith_mean_round_{}.csv".format(args.num_folds, num_round), index = False)

        all_aucs.extend(best_aucs)
        if num_round == 0:
            train_metas = train_meta
            test_metas = sub[LABELS]
        else:
            train_metas += train_meta
            test_metas += sub[LABELS]

    if args.num_rounds > 1 and args.save_prediction:
        train_metas /= args.num_rounds
        test_metas /= args.num_rounds
        pd.DataFrame(train_metas, columns = LABELS).to_csv(out_dir + "/train_metas_all.csv", index = False)
        test_metas.to_csv(out_dir + "/test_metas_all.csv", index = False)

        print("mean of all aucs {} -- std {}".format(np.mean(all_aucs), np.std(all_aucs)))


if __name__ == '__main__':
    main()
