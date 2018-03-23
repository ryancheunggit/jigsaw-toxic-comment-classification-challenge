# taken from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50827

import pandas as pd
import numpy as np
import sys
from scipy.stats import ks_2samp

first_file = sys.argv[1]
second_file = sys.argv[2]

def corr(first_file, second_file):
    # assuming first column is `class_name_id`
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    pcs, kcs, scs, ks, kp = [], [], [], [], []

    for class_name in class_names:
        # all correlations
        print('\n Class: %s' % class_name)
        pc = first_df[class_name].corr(second_df[class_name], method='pearson')
        pcs.append(pc)
        print(' Pearson\'s correlation score: %0.6f' % pc)
        kc = first_df[class_name].corr(second_df[class_name], method='kendall')
        kcs.append(kc)
        print(' Kendall\'s correlation score: %0.6f' % kc)
        sc = first_df[class_name].corr(second_df[class_name], method='spearman')
        scs.append(sc)
        print(' Spearman\'s correlation score: %0.6f' % sc)
        ks_stat, p_value = ks_2samp(first_df[class_name].values, second_df[class_name].values)
        ks.append(ks_stat)
        kp.append(p_value)
        print(' Kolmogorov-Smirnov test:    KS-stat = %.6f    p-value = %.3e\n'% (ks_stat, p_value))
    print(' Pearson correlation avg: %0.6f std: %0.6f' % (np.mean(pcs), np.std(pcs)))
    print(' Kendall correlation avg: %0.6f std: %0.6f' % (np.mean(kcs), np.std(kcs)))
    print(' Spearman correlation avg: %0.6f std: %0.6f' % (np.mean(scs), np.std(scs)))
    print(' Kolmogorov-Smirnov stat avg: %0.6f std: %0.6f' % (np.mean(ks), np.std(ks)))
    print(' Kolmogorov-Smirnov p avg: %0.6f std: %0.6f' % (np.mean(kp), np.std(kp)))

corr(first_file, second_file)
