#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import string
import re
import pandas as pd
import numpy as np
from .constants import *
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics import log_loss

tqdm.pandas(desc="progress")


def load_data(processed = False):
    """
        load training and testing data, get label and label distribution
        :return
            train: df with id, comments and targets(binary)
            test: df with id and comments
            y: labels
            y_label_dist: unique combination of labels as strings, will be used to do cv splits
        # processed data got from https://www.kaggle.com/fizzbuzz/cleaned-toxic-comments
    """
    if processed:
        train = pd.read_csv("../input/train_preprocessed.csv")
        test = pd.read_csv("../input/test_preprocessed.csv")
        train.drop(labels = ['set', 'toxicity'], axis = 1, inplace = True)
        test.drop(labels = ['set', 'toxicity'], axis = 1, inplace = True)
        train[LABELS] = train[LABELS].astype(int)
        test.comment_text = test.comment_text.astype(str)
    else:
        train = pd.read_csv("../input/train.csv")
        test = pd.read_csv("../input/test.csv")
    y = train[LABELS].values
    y_label_dist = train.progress_apply(lambda row: "-".join([str(label) for label in  row[LABELS]]), axis = 1)
    return train, test, y, y_label_dist


def deduuup_text(text):
    """
        https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50942
        this function processes word like leeroooy Jeeenkinnnnns to leeroy jeenkins
    """
    for ch in string.ascii_lowercase[:27]:
        if ch in text:
            template=r"("+ch+")\\1{2,}"
            text = re.sub(template, ch, text)
    return text


def process_sentences(
        sentences,
        word_idx_mapping,
        tokenizer = 'treebank_punkt'
    ):
    """
        Tokenize sentences, create/update word:index mapping
        :param
            sentences: list of comment strings
            word_idx_mapping: dictionary mapping word to its idx number
            tokenizer: tokenizer to use
        :return
            processed_sentences: list of lists of word idx numbers
            word_idx_mapping: updated dictionary mapping word to its idx number
    """
    tokenizers = {
        'treebank_punkt': word_tokenize,
        'punkt': sent_tokenize,
        'word_seq': text_to_word_sequence
    }
    tokenize = tokenizers[tokenizer]
    
    processed_sentences = []
    for sentence in tqdm(sentences):
        # sentence = deduuup_text(sentence)
        tokens = tokenize(sentence)
        processed_sentence = []
        for token in tokens:
            word = token.lower()
            if word not in word_idx_mapping:
                word_idx_mapping[word] = len(word_idx_mapping)
            word_idx = word_idx_mapping[word]
            processed_sentence.append(word_idx)
        processed_sentences.append(processed_sentence)
    return processed_sentences, word_idx_mapping

def load_embeddings(word_idx_mapping, embedding_file):
    """
        load embedding vector and create a word to embedding indicies mapping

        :param
            word_idx_mapping: the word to indices mapping for the sentences
            tokenizer: tokenizer to use
        :return
            processed_sentences: list of lists of word idx numbers
            word_id_mapping: updated dictionary mapping word to its idx number
    """
    embedding_file_func_mapping = {
            'crawl-300d-2M.vec': load_fasttext_embeddings,
            'glove.840B.300d.txt': load_glove_embeddings,
            'GoogleNews-vectors-negative300.bin': load_word2vec_embeddings
        }

    embedding_word_idx_mapping, embedding_vectors = embedding_file_func_mapping[embedding_file](word_idx_mapping)
    return embedding_word_idx_mapping, embedding_vectors


def load_glove_embeddings(word_idx_mapping):
    assert os.path.exists("../input/glove.840B.300d.txt"), "glove embedding file not exist"
    embedding_word_idx_mapping = {}
    embedding_vectors = []
    with open("../input/glove.840B.300d.txt") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            line = line.split("\n")[0]
            data = line.split(" ")
            word = data[0]
            if word in word_idx_mapping:
                embedding = np.array([float(num) for num in data[1:]])
                embedding_vectors.append(embedding)
                embedding_word_idx_mapping[word] = len(embedding_word_idx_mapping)
    embedding_word_idx_mapping[WORD_WITH_NO_EMBEDDING] = len(embedding_word_idx_mapping)
    embedding_vectors.append(np.zeros(300))
    embedding_word_idx_mapping[SENTENCE_END] = len(embedding_word_idx_mapping)
    embedding_vectors.append( -1 * np.ones(300))
    embedding_vectors = np.array(embedding_vectors)
    return embedding_word_idx_mapping, embedding_vectors


def load_fasttext_embeddings(word_idx_mapping):
    assert os.path.exists("../input/crawl-300d-2M.vec"), "fastText embedding file not exist"
    embedding_word_idx_mapping = {}
    embedding_vectors = []
    with open("../input/crawl-300d-2M.vec") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            if idx == 0:
                continue
            line = line.split("\n")[0]
            data = line.split(" ")
            word = data[0]
            if word in word_idx_mapping:
                embedding = np.array([float(num) for num in data[1:-1]])
                embedding_vectors.append(embedding)
                embedding_word_idx_mapping[word] = len(embedding_word_idx_mapping)
    embedding_word_idx_mapping[WORD_WITH_NO_EMBEDDING] = len(embedding_word_idx_mapping)
    embedding_vectors.append(np.zeros(300))
    embedding_word_idx_mapping[SENTENCE_END] = len(embedding_word_idx_mapping)
    embedding_vectors.append( -1 * np.ones(300))
    embedding_vectors = np.array(embedding_vectors)
    return embedding_word_idx_mapping, embedding_vectors


def load_word2vec_embeddings(word_idx_mapping):
    import gensim.models.keyedvectors as word2vec
    assert os.path.exists("../input/GoogleNews-vectors-negative300.bin"), "word2vec embedding file not exist"
    embedding_word_idx_mapping = {}
    embedding_vectors = []
    word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/GoogleNews-vectors-negative300.bin",binary=True)
    for word in tqdm(word2vecDict.wv.vocab):
        if word in word_idx_mapping:
            embedding_word_idx_mapping[word] = len(embedding_word_idx_mapping)
            embedding_vectors.append(word2vecDict.word_vec(word))
    embedding_word_idx_mapping[WORD_WITH_NO_EMBEDDING] = len(embedding_word_idx_mapping)
    embedding_vectors.append(np.zeros(300))
    embedding_word_idx_mapping[SENTENCE_END] = len(embedding_word_idx_mapping)
    embedding_vectors.append( -1 * np.ones(300))
    embedding_vectors = np.array(embedding_vectors)
    return embedding_word_idx_mapping, embedding_vectors


def word_indices_to_embedding_indices(
        processed_sentences,
        idx_word_mapping,
        embedding_word_idx_mapping,
        max_sentence_length,
        pad_unknown = True
    ):
    """
        convert word_indices to embedding_indices

        :param
            processed_sentences: list of sequences word indices
            idx_word_mapping: mapping from word indices to actual word # look up word from seq
            embedding_word_idx_mapping: mapping form word to embedding vector indices # look up wv
            max_sentence_length: maximum number of words per sentence
            pad_unknown: wheter to encode unknown with zero vector

        :return
            list of lists of loaded word vectors
    """
    sentences_embedded = []
    for sentence in tqdm(processed_sentences):
        sentence_embedding_indices = []
        for word_idx in sentence:
            word = idx_word_mapping[word_idx]
            embedding_idx = embedding_word_idx_mapping.get(word, len(embedding_word_idx_mapping) - 2)
            if (not pad_unknown) and (word not in embedding_word_idx_mapping):
                continue
            sentence_embedding_indices.append(embedding_idx)
        if len(sentence_embedding_indices) >= max_sentence_length:
            # crop sentence
            sentence_embedding_indices = sentence_embedding_indices[:max_sentence_length]
        else:
            # pad sentence
            sentence_embedding_indices += [len(embedding_word_idx_mapping) - 1] *\
                                          (max_sentence_length - len(sentence_embedding_indices))
        sentences_embedded.append(sentence_embedding_indices)
    return sentences_embedded

def str2bool(v):
    import argparse
    """ used to parse string argument true and false to bool in argparse """
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def avg_log_loss(y_true, y_preds):
    y_preds = y_preds.astype("float64")
    losses = [log_loss(y_true[:, j], y_preds[:, j]) for j in range(6)]
    avg_losses = np.mean(losses)
    return avg_losses
