#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from .constants import *
from datetime import datetime
from util import load_data, str2bool
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
tqdm.pandas(desc="progress")


lgbm_params = {
        'all_models': { # 0.9922771050991881
            'toxic': { # 0.9885532212790336
                'learning_rate': 0.06, 'n_estimators': 120, 'min_child_samples': 40, 'num_leaves': 18,
                'colsample_bytree': 0.3, 'subsample': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 0.5
                },
            'severe_toxic': { # 0.9922408922400298
                'learning_rate': 0.05, 'n_estimators': 160, 'min_child_samples': 80, 'num_leaves': 15,
                'colsample_bytree': 0.3, 'subsample': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 0.3
                },
            'obscene': { # 0.995680861094397
                'learning_rate': 0.03, 'n_estimators': 140, 'min_child_samples': 90, 'num_leaves': 18,
                'colsample_bytree': 0.4, 'subsample': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 0.5
                },
            'threat': { # 0.9938790013464035
                'learning_rate': 0.07, 'n_estimators': 180, 'min_child_samples': 90, 'num_leaves': 15,
                'colsample_bytree': 0.6, 'subsample': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 0.7
                },
            'insult': { # 0.9903820705298804
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 80, 'num_leaves': 15,
                'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.3
                },
            'identity_hate': { # 0.9929265841053845
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 80, 'num_leaves': 15,
                'colsample_bytree': 0.3, 'subsample': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 0.2
                }
        },
        'model_bench': { # 0.9924219564254093
            'toxic': { # 0.9884738285743253
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 60, 'num_leaves': 18,
                'colsample_bytree': 0.4, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0.2
                },
            'severe_toxic': { # 0.9921760613562598
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 60, 'num_leaves': 27,
                'colsample_bytree': 0.6, 'subsample': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.7
                },
            'obscene': { # 0.9957039633662825
                'learning_rate': 0.05, 'n_estimators': 140, 'min_child_samples': 60, 'num_leaves': 18,
                'colsample_bytree': 0.5, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.7
                },
            'threat': { # 0.9951981320611659
                'learning_rate': 0.05, 'n_estimators': 120, 'min_child_samples': 40, 'num_leaves': 18,
                'colsample_bytree': 0.4, 'subsample': 1, 'reg_alpha': 0.5, 'reg_lambda': 0.3
                },
            'insult': { # 0.9903831509852782
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 60, 'num_leaves': 18,
                'colsample_bytree': 0.5, 'subsample': 0.8, 'reg_alpha': 0.3, 'reg_lambda': 0.7
                },
            'identity_hate': { # 0.9925966022091445
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 60, 'num_leaves': 18,
                'colsample_bytree': 0.4, 'subsample': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 0.7
                }
        },
        'de_corred_models': { # 0.9922108802620007
            'toxic': { # 0.9884339880176795
                'learning_rate': 0.03, 'n_estimators': 160, 'min_child_samples': 30, 'num_leaves': 18,
                'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.3
                },
            'severe_toxic': { # 0.9921487206416756
                'learning_rate': 0.05, 'n_estimators': 160, 'min_child_samples': 90, 'num_leaves': 15,
                'colsample_bytree': 0.3, 'subsample': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.2
                },
            'obscene': { # 0.9956632949593645
                'learning_rate': 0.03, 'n_estimators': 140, 'min_child_samples': 60, 'num_leaves': 15,
                'colsample_bytree': 0.3, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.2
                },
            'threat': { # 0.9941969488469569
                'learning_rate': 0.06, 'n_estimators': 180, 'min_child_samples': 30, 'num_leaves': 15,
                'colsample_bytree': 0.5, 'subsample': 0.9, 'reg_alpha': 0.3, 'reg_lambda': 0.9
                },
            'insult': { # 0.9903199328940103
                'learning_rate': 0.03, 'n_estimators': 100, 'min_child_samples': 90, 'num_leaves': 30,
                'colsample_bytree': 0.4, 'subsample': 0.7, 'reg_alpha': 0.3, 'reg_lambda': 0.9
                },
            'identity_hate':{ # 0.9925023962123172
                'learning_rate': 0.05, 'n_estimators': 180, 'min_child_samples': 80, 'num_leaves': 15,
                'colsample_bytree': 0.4, 'subsample': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.2
                }
        }
    }

xgbm_params = {
        'all_models':{
            'toxic':{ # 0.9883646210560985
                'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 1, 'max_depth': 2,
                'subsample': 0.8, 'colsample_bytree': 1, 'reg_alpha': 0, 'reg_lambda': 1
                },
            'severe_toxic': { # 0.9923602463534126
                'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 5, 'max_depth': 3,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 1
                },
            'obscene':{ # 0.9956809352739643
                'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 3, 'max_depth': 2,
                'subsample': 0.8, 'colsample_bytree': 1, 'reg_alpha': 0.1, 'reg_lambda': 1
                },
            'threat':{ # 0.9951609378981362
                'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 5, 'max_depth': 3,
                'subsample': 1, 'colsample_bytree': 1, 'reg_alpha': 0.1, 'reg_lambda': 1
                },
            'insult':{ # 0.9903081803936655
                'learning_rate': 0.05, 'n_estimators': 140, 'min_child_weight': 5, 'max_depth': 3,
                'subsample': 0.8, 'colsample_bytree': 1, 'reg_alpha': 0.2, 'reg_lambda': 0.9
                },
            'identity_hate': { # 0.9924017269427837
                'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 5, 'max_depth': 2,
                'subsample': 0.7 , 'colsample_bytree': 1 , 'reg_alpha': 0 , 'reg_lambda': 0.9
                }
        }
    }

lr_params = { # 0.9907433586233355
        'toxic': {'C': 0.1, 'solver': 'newton-cg', 'tol': 0.01},         # 0.9871983550069076
        'severe_toxic': {'C': 0.05, 'solver': 'newton-cg', 'tol': 0.01}, # 0.9918178163403378
        'obscene': {'C': 0.05, 'solver': 'newton-cg', 'tol': 0.01},      # 0.9949293191157444
        'threat': {'C': 1, 'solver': 'lbfgs', 'tol': 0.01},              # 0.9907905802530971
        'insult': {'C': 0.05, 'solver': 'newton-cg', 'tol': 0.01},       # 0.9896128827866938
        'identity_hate': {'C': 0.05, 'solver': 'newton-cg', 'tol': 0.01} # 0.9901111982372319
    }

# all the model i saved
all_models = [
    "e-fastText-m-singlegru-o-loss-l-300-t-word_seq-a-False-k-5-d-0.5-ed-True"
    , "e-fastText-m-singlegru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.5-ed-True"
    , "e-glove-m-singlegru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.5-ed-True"
    , "e-glove-m-singlegru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.5-ed-True"
    , "e-fastText-m-singlegru-o-loss-l-300-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-fastText-m-singlegru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-glove-m-singlegru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-glove-m-singlegru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , 'e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p'
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-duallstm-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-duallstm-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-lstmpool-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False
    , "e-fastText-m-lstmpool-o-loss-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p
    , "e-glove-m-lstmpool-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-lstmpool-o-loss-l-400-t-word_seq-a-False-k-5-d-0.1-ed-True"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.1-ed-True"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-1dcnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.3-ed-False"
    , "e-glove-m-1dcnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.3-ed-False"
    , "e-glove-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.3-ed-False-p"
    , "e-glove-m-1dcnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.3-ed-True-p"
    , "e-fastText-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-fastText-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-glove-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True-p"
    , "e-fastText-m-2dcnn-o-auc-l-200-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-2dcnn-o-loss-l-200-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-2dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-2dcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-crnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-crnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-grucnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-grucnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-grucnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-grucnn-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-grucnn-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "log_reg-w-None-3-c-None-6-C-4-k-5"
    , "log_reg-w-None-3-c-None-6-C--1-k-5"
    , "log_reg-w-25000-2-c-60000-3-C--1.0-k-5"
    , "log_reg-w-None-3-c-None-6-C--1.0-k-5-p"
    , "fm_ftrl-w-20000-3-c-50000-6"
    , "fm_ftrl-w-20000-2-c-50000-3"
    , "fm_ftrl-w-300000-2-c-60000-3-p"
    ]

# the models i used to experiment stacking before the final weekend, got 0.9871 with lgb on this set
model_bench = [
    "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , 'e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.5-ed-True-p
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-lstmpool-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False
    , "e-fastText-m-lstmpool-o-loss-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p
    , "e-glove-m-lstmpool-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-lstmpool-o-loss-l-400-t-word_seq-a-False-k-5-d-0.1-ed-True"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.1-ed-True"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-1dcnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.3-ed-False"
    , "e-glove-m-1dcnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.3-ed-False"
    , "e-glove-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.3-ed-False-p"
    , "e-glove-m-1dcnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.3-ed-True-p"
    , "e-fastText-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-fastText-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-glove-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True-p"
    , "e-fastText-m-2dcnn-o-auc-l-200-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-2dcnn-o-loss-l-200-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-2dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-2dcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-crnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-crnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-grucnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-grucnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-grucnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-grucnn-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-grucnn-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "log_reg-w-None-3-c-None-6-C-4-k-5"
    , "log_reg-w-None-3-c-None-6-C--1-k-5"
    , "log_reg-w-25000-2-c-60000-3-C--1.0-k-5"
    , "log_reg-w-None-3-c-None-6-C--1.0-k-5-p"
    , "fm_ftrl-w-20000-3-c-50000-6"
    , "fm_ftrl-w-20000-2-c-50000-3"
    , "fm_ftrl-w-300000-2-c-60000-3-p"
    ]

# the set of models with no model pair with > 0.96 pearson correlation
de_corred_models = [
      "e-fastText-m-singlegru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.5-ed-True"
    , "e-glove-m-singlegru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.5-ed-True"
    , "e-fastText-m-singlegru-o-loss-l-300-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-glove-m-singlegru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-dualgru-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-dualgru-o-loss-l-500-t-word_seq-a-False-k-5-d-0.5-ed-True-p"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-dualgru-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-dualgru-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-duallstm-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-duallstm-o-auc-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-fastText-m-lstmpool-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-lstmpool-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-glove-m-lstmpool-o-auc-l-400-t-word_seq-a-False-k-5-d-0.1-ed-True"
    , "e-fastText-m-1dcnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.3-ed-False"
    , "e-glove-m-1dcnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.3-ed-False"
    , "e-glove-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-1dcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.3-ed-False-p"
    , "e-glove-m-1dcnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.3-ed-True-p"
    , "e-fastText-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-fastText-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True"
    , "e-glove-m-dpcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-glove-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-dpcnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.25-ed-True-p"
    , "e-fastText-m-2dcnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-fastText-m-crnn-o-loss-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-crnn-o-auc-l-300-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-crnn-o-loss-l-400-t-word_seq-a-False-k-5-d-0.2-ed-False"
    , "e-fastText-m-grucnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "e-fastText-m-grucnn-o-auc-l-400-t-word_seq-a-False-k-5-d-0.2-ed-True-p"
    , "e-glove-m-grucnn-o-loss-l-500-t-word_seq-a-False-k-5-d-0.2-ed-True"
    , "log_reg-w-None-3-c-None-6-C-4-k-5"
    , "log_reg-w-None-3-c-None-6-C--1.0-k-5-p"
    , "fm_ftrl-w-20000-2-c-50000-3"
    , "fm_ftrl-w-300000-2-c-60000-3-p"
    ]

def main():
    parser = argparse.ArgumentParser(description='training nn and make predictions')
    parser.add_argument('--dataset', type = str, default = "all_models", help = 'dataset to use for layer2 stacking')
    parser.add_argument('--mode', type = str, default = "OOF", help = 'do cv tuning or oof generation')
    parser.add_argument('--model', type = str, default = "xgb", help = 'what model to use for stacking')
    parser.add_argument('--save_flag', type = str, default = "0", help = 'versioning flag')
    parser.add_argument('--save_prediction', type = str2bool, default = "True", help = 'save prediciton or not')
    args = parser.parse_args()
    print(args)

    models = all_models
    if args.dataset == "all_models":
        models = all_models
    elif args.dataset == 'model_bench':
        models = model_bench
    elif args.dataset == 'de_corred_models':
        models = de_corred_models

    train, test, y, y_label_dist = load_data(processed = True)
    sub = pd.read_csv("../input/sample_submission.csv")

    # some ad hoc features
    train['comment_text'].fillna("__UNKNOWN__", inplace = True)
    test['comment_text'].fillna("__UNKNOWN__", inplace = True)
    train['num_words'] = train.comment_text.str.count('\S+')
    test['num_words'] = test.comment_text.str.count('\S+')
    train['num_comas'] = train.comment_text.str.count('\.')
    test['num_comas'] = test.comment_text.str.count('\.')
    train['num_bangs'] = train.comment_text.str.count('\!')
    test['num_bangs'] = test.comment_text.str.count('\!')
    train['num_quotas'] = train.comment_text.str.count('\"')
    test['num_quotas'] = test.comment_text.str.count('\"')
    train['avg_word'] = train.comment_text.str.len() / (1 + train.num_words)
    test['avg_word'] = test.comment_text.str.len() / (1 + test.num_words)
    sent_analyzer = SentimentIntensityAnalyzer()
    train['sentiments'] = train.comment_text.progress_map(lambda text: sent_analyzer.polarity_scores(text)['compound'])
    test['sentiments'] = test.comment_text.progress_map(lambda text: sent_analyzer.polarity_scores(text)['compound'])
    META_FEATURES = [ 'num_words', 'num_comas', 'num_bangs', 'num_quotas', 'avg_word', 'sentiments']

    # read in oof predictions from layer1
    train_features = pd.concat([
        pd.read_csv(inp)[LABELS]
        for inp in ["../models/{}/train_meta_probs_round_0.csv".format(model)
        for model in models]]
    , axis = 1)
    train_features.columns = ['_'.join([label, str(i + 1)]) for i in range(len(models)) for label in LABELS]

    train_features = pd.concat([train_features, train[META_FEATURES]], axis = 1)

    # read in avg test predicitons from layer 1
    test_features = pd.concat([
        pd.read_csv(inp)[LABELS]
        for inp in ["../models/{}/test_probs_5_bag_arith_mean_round_0.csv".format(model)
        for model in models]]
    , axis = 1)
    test_features.columns = ['_'.join([label, str(i + 1)]) for i in range(len(models)) for label in LABELS]
    test_features = pd.concat([test_features, test[META_FEATURES]], axis = 1)

    # === cv splits and place holders
    # I am reusing the same split from layer 1
    splitter = StratifiedKFold(n_splits = 5, shuffle = True, random_state = CV_SPLIT_SEED)
    folds = list(splitter.split(train_features, y_label_dist))

    if args.mode == 'CV':
        if args.model == 'lgb':
            lgb_params = {}
            aucs = []

            # per label cv tuning
            for idx, label in enumerate(LABELS):
                print("idx {} label {} started cv at time {}".format(idx, label, datetime.now()))
                current_param_set = {}
                lgb_model = lgb.LGBMClassifier(objective = 'binary', n_jobs = 8, class_weight = 'balanced')
                for param_pairs in [
                        {'learning_rate': [0.02, 0.03, 0.05, 0.06, 0.07],
                         'n_estimators': [100, 120, 140, 160, 180]},
                        {'num_leaves': [15, 18, 24, 27, 30],
                         'min_child_samples': [30, 40, 60, 80, 90]},
                        {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
                         'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7]},
                        {'reg_alpha': [0, 0.1, 0.2, 0.3, 0.5],
                         'reg_lambda': [0.2, 0.3, 0.5, 0.7, 0.9]}
                    ]:

                    grid_search = GridSearchCV(
                            estimator = lgb_model
                            , param_grid = param_pairs
                            , scoring = 'roc_auc'
                            , n_jobs = 1
                            , cv = folds
                            , refit = True
                            , verbose = 1
                            , return_train_score = True
                        )
                    results = grid_search.fit(train_features, y[:, idx])
                    current_param_set.update(results.best_params_)
                    lgb_model = results.best_estimator_
                    print(results.best_score_)
                    print(lgb_model)
                sub[label] = lgb_model.predict_proba(test_features)[:,1]
                print(results.best_score_)
                print(current_param_set)
                aucs.append(results.best_score_)
                lgb_params[label] = current_param_set

            print(np.mean(aucs))
            if args.save_prediction:
                sub.to_csv("lgb_stacker_ver{}.csv".format(args.save_flag), index = False)

        if args.model == 'lr':
            log_reg_params = {}
            aucs = []

            # per label cv tuning
            for idx, label in enumerate(LABELS):
                print("idx {} label {} started cv at time {}".format(idx, label, datetime.now()))
                log_reg = LogisticRegression(fit_intercept = True, penalty = 'l2', class_weight = 'balanced')
                param_grid = {
                        'C':   [0.001, 0.05, 0.1, 1, 2, 10],
                        'tol': [0.01],
                        'solver': ['lbfgs', 'newton-cg']
                    }

                grid_search = GridSearchCV(
                        estimator = log_reg
                        , param_grid = param_grid
                        , scoring = 'roc_auc'
                        , n_jobs = 8
                        , cv = folds
                        , refit = True
                        , verbose = 1
                        , return_train_score = True
                    )
                results = grid_search.fit(train_features, y[:, idx])
                log_reg = results.best_estimator_
                print(results.best_score_)
                print(results.best_params_)

                sub[label] = log_reg.predict_proba(test_features)[:,1]
                aucs.append(results.best_score_)
                log_reg_params[label] = results.best_params_

            print(np.mean(aucs))
            if args.save_prediction:
                sub.to_csv("log_reg_stacker_ver{}.csv".format(args.save_flag), index = False)

        if args.model == 'xgb':
            xgb_params = {}
            aucs = []
            test_probs = []
            # per label cv tuning
            for idx, label in enumerate(LABELS):
                print("idx {} label {} started cv at time {}".format(idx, label, datetime.now()))
                current_param_set = {}
                xgb_model = xgb.XGBClassifier(objective = 'binary:logistic', n_jobs = 8, class_weight = 'balanced')
                for param_pairs in [
                        {'learning_rate': [0.04, 0.05, 0.06]},
                        {'n_estimators': [120, 140, 150, 160]},
                        {'max_depth': [2,3,4]},
                        {'min_child_weight': [1,3,5]},
                        {'subsample': [0.8, 1]},
                        {'colsample_bytree': [0.8, 1]},
                        {'reg_alpha': [0, 0.1]},
                        {'reg_lambda': [0.9, 1]}
                    ]:
                    # print(np.mean(
                    #   cross_val_score(
                    #       xgb_model,
                    #       train_features,
                    #       y[:, idx],
                    #       cv = folds,
                    #       scoring = 'roc_auc',
                    #       verbose = 2
                    #   )))
                    grid_search = GridSearchCV(
                            estimator = xgb_model
                            , param_grid = param_pairs
                            , scoring = 'roc_auc'
                            , n_jobs = 1
                            , cv = folds
                            , refit = True
                            , verbose = 2
                            , return_train_score = True
                        )
                    results = grid_search.fit(train_features, y[:, idx])
                    current_param_set.update(results.best_params_)
                    xgb_model = results.best_estimator_
                    print(results.best_score_)
                    print(xgb_model)
                sub[label] = xgb_model.predict_proba(test_features)[:,1]
                print(results.best_score_)
                print(current_param_set)
                aucs.append(results.best_score_)
                xgb_params[label] = current_param_set

            print(np.mean(aucs))
            if args.save_prediction:
                sub.to_csv("xgb_stacker_ver{}.csv".format(args.save_flag), index = False)


    if args.mode == 'OOF':
        if args.model == 'lgb':
            model_params = lgbm_params[args.dataset]
            train_metas = np.zeros(y.shape)
            aucs = []
            losses = []
            test_probs = []
            classifiers = {}
            aucs_per_label = {}

            for fold_num, [train_indices, valid_indices] in enumerate(folds):
                print("=== fitting fold {} datetime {} ===".format(fold_num, datetime.now()))
                x_train, x_valid = train_features.values[train_indices,:], train_features.values[valid_indices,:]
                y_train, y_valid = y[train_indices], y[valid_indices]

                valid_preds = np.zeros(y_valid.shape)
                test_preds = np.zeros((test_features.shape[0], len(LABELS)))

                for idx, label in enumerate(LABELS):
                    print("fitting lightgbm for label {} at time {}".format(label, datetime.now()))
                    classifier = "fold_{}_{}".format(fold_num, label)
                    classifiers[classifier] = lgb.LGBMClassifier(
                        objective = 'binary',
                        n_jobs = 8,
                        class_weight = 'balanced',
                        learning_rate = model_params[label]['learning_rate'],
                        num_leaves = model_params[label]['num_leaves'],
                        n_estimators = model_params[label]['n_estimators'],
                        min_child_samples = model_params[label]['min_child_samples'],
                        subsample = model_params[label]['subsample'],
                        colsample_bytree = model_params[label]['colsample_bytree'],
                        reg_alpha = model_params[label]['reg_alpha'],
                        reg_lambda = model_params[label]['reg_lambda']
                    )
                    classifiers[classifier].fit(x_train, y_train[:, idx])
                    valid_preds[:, idx] = classifiers[classifier].predict_proba(x_valid)[:, 1]
                    test_preds[:, idx] = classifiers[classifier].predict_proba(test_features)[:, 1]
                    auc_score = roc_auc_score(y_valid[:, idx], valid_preds[:, idx])

                    if label not in aucs_per_label:
                        aucs_per_label[label] = [auc_score]
                    else:
                        aucs_per_label[label].append(auc_score)

                train_metas[valid_indices] = valid_preds
                test_probs.append(test_preds)
                auc_score = roc_auc_score(y_valid, valid_preds)
                log_loss_score = log_loss(y_valid, valid_preds)

                print("validation auc {} log loss {}".format(auc_score, log_loss_score))
                aucs.append(auc_score)
                losses.append(log_loss_score)

            aaa = []
            for label in aucs_per_label:
                print(np.mean(aucs_per_label[label]))
                aaa.append(np.mean(aucs_per_label[label]))
            print(np.mean(aaa))

            print("mean auc score: {} - std {} , mean log loss score: {} - std {}".format(
                    np.mean(aucs), np.std(aucs), np.mean(losses), np.std(losses)
                ))

            out_dir = '../models/layer2/{}-{}-{}'.format(args.model, args.dataset, args.save_flag)
            try:
                os.mkdir(out_dir)
            except:
                print("path exists or failed to create")

            pd.DataFrame(train_metas, columns = LABELS).to_csv(out_dir + "/train_meta_probs_round_0.csv", index = False)

            sub[LABELS] = np.zeros(sub[LABELS].shape)
            for i in range(5):
                sub[LABELS] += test_probs[i]
            sub[LABELS] /= 5
            sub.to_csv(out_dir + "/test_probs_5_bag_arith_mean_round_0.csv", index = False)

        if args.model == 'lr':
            train_metas = np.zeros(y.shape)
            aucs = []
            losses = []
            test_probs = []
            classifiers = {}
            aucs_per_label = {}

            for fold_num, [train_indices, valid_indices] in enumerate(folds):
                print("=== fitting fold {} datetime {} ===".format(fold_num, datetime.now()))
                x_train, x_valid = train_features.values[train_indices,:], train_features.values[valid_indices,:]
                y_train, y_valid = y[train_indices], y[valid_indices]

                valid_preds = np.zeros(y_valid.shape)
                test_preds = np.zeros((test_features.shape[0], len(LABELS)))

                for idx, label in enumerate(LABELS):
                    print("fitting logistic regression for label {} at time {}".format(label, datetime.now()))
                    classifier = "fold_{}_{}".format(fold_num, label)
                    classifiers[classifier] = LogisticRegression(
                        fit_intercept = True,
                        penalty = 'l2',
                        class_weight = 'balanced',
                        C = lr_params[label]['C'],
                        tol = lr_params[label]['tol'],
                        solver = lr_params[label]['solver'],
                    )
                    classifiers[classifier].fit(x_train, y_train[:, idx])
                    valid_preds[:, idx] = classifiers[classifier].predict_proba(x_valid)[:, 1]
                    test_preds[:, idx] = classifiers[classifier].predict_proba(test_features)[:, 1]
                    auc_score = roc_auc_score(y_valid[:, idx], valid_preds[:, idx])

                    if label not in aucs_per_label:
                        aucs_per_label[label] = [auc_score]
                    else:
                        aucs_per_label[label].append(auc_score)

                train_metas[valid_indices] = valid_preds
                test_probs.append(test_preds)
                auc_score = roc_auc_score(y_valid, valid_preds)
                log_loss_score = log_loss(y_valid, valid_preds)

                print("validation auc {} log loss {}".format(auc_score, log_loss_score))
                aucs.append(auc_score)
                losses.append(log_loss_score)

            aaa = []
            for label in aucs_per_label:
                print(np.mean(aucs_per_label[label]))
                aaa.append(np.mean(aucs_per_label[label]))
            print(np.mean(aaa))

            print("mean auc score: {} - std {} , mean log loss score: {} - std {}".format(
                    np.mean(aucs), np.std(aucs), np.mean(losses), np.std(losses)
                ))

            out_dir = '../models/layer2/{}-{}-{}'.format(args.model, args.dataset, args.save_flag)
            try:
                os.mkdir(out_dir)
            except:
                print("path exists or failed to create")

            pd.DataFrame(train_metas, columns = LABELS).to_csv(out_dir + "/train_meta_probs_round_0.csv", index = False)

            sub[LABELS] = np.zeros(sub[LABELS].shape)
            for i in range(5):
                sub[LABELS] += test_probs[i]
            sub[LABELS] /= 5
            sub.to_csv(out_dir + "/test_probs_5_bag_arith_mean_round_0.csv", index = False)

        if args.model == 'xgb':
            test_features = test_features.values
            model_params = xgbm_params[args.dataset]
            train_metas = np.zeros(y.shape)
            aucs = []
            losses = []
            test_probs = []
            classifiers = {}
            aucs_per_label = {}

            for fold_num, [train_indices, valid_indices] in enumerate(folds):
                print("=== fitting fold {} datetime {} ===".format(fold_num, datetime.now()))
                x_train, x_valid = train_features.values[train_indices,:], train_features.values[valid_indices,:]
                y_train, y_valid = y[train_indices], y[valid_indices]

                valid_preds = np.zeros(y_valid.shape)
                test_preds = np.zeros((test_features.shape[0], len(LABELS)))

                for idx, label in enumerate(LABELS):
                    print("fitting xgboost for label {} at time {}".format(label, datetime.now()))
                    classifier = "fold_{}_{}".format(fold_num, label)
                    classifiers[classifier] = xgb.XGBClassifier(
                        objective = 'binary:logistic',
                        n_jobs = 8,
                        class_weight = 'balanced',
                        learning_rate = model_params[label]['learning_rate'],
                        n_estimators = model_params[label]['n_estimators'],
                        max_depth = model_params[label]['max_depth'],
                        min_child_weight = model_params[label]['min_child_weight'],
                        subsample = model_params[label]['subsample'],
                        colsample_bytree = model_params[label]['colsample_bytree'],
                        reg_alpha = model_params[label]['reg_alpha'],
                        reg_lambda = model_params[label]['reg_lambda']
                    )
                    classifiers[classifier].fit(x_train, y_train[:, idx])
                    valid_preds[:, idx] = classifiers[classifier].predict_proba(x_valid)[:, 1]
                    test_preds[:, idx] = classifiers[classifier].predict_proba(test_features)[:, 1]
                    auc_score = roc_auc_score(y_valid[:, idx], valid_preds[:, idx])
                    gc.collect()
                    if label not in aucs_per_label:
                        aucs_per_label[label] = [auc_score]
                    else:
                        aucs_per_label[label].append(auc_score)

                train_metas[valid_indices] = valid_preds
                test_probs.append(test_preds)
                auc_score = roc_auc_score(y_valid, valid_preds)
                log_loss_score = log_loss(y_valid, valid_preds)

                print("validation auc {} log loss {}".format(auc_score, log_loss_score))
                aucs.append(auc_score)
                losses.append(log_loss_score)

            aaa = []
            for label in aucs_per_label:
                print(np.mean(aucs_per_label[label]))
                aaa.append(np.mean(aucs_per_label[label]))
            print(np.mean(aaa))

            print("mean auc score: {} - std {} , mean log loss score: {} - std {}".format(
                    np.mean(aucs), np.std(aucs), np.mean(losses), np.std(losses)
                ))

            out_dir = '../models/layer2/{}-{}-{}'.format(args.model, args.dataset, args.save_flag)
            try:
                os.mkdir(out_dir)
            except:
                print("path exists or failed to create")

            pd.DataFrame(train_metas, columns = LABELS).to_csv(out_dir + "/train_meta_probs_round_0.csv", index = False)

            sub[LABELS] = np.zeros(sub[LABELS].shape)
            for i in range(5):
                sub[LABELS] += test_probs[i]
            sub[LABELS] /= 5
            sub.to_csv(out_dir + "/test_probs_5_bag_arith_mean_round_0.csv", index = False)

if __name__ == '__main__':
    main()
