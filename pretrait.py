from __future__ import print_function

import numpy as np
import re
import itertools
from collections import Counter
from nltk.stem import *
from nltk.corpus import stopwords
import pandas as pd
import collections
import _pickle as cPickle
import random
import tensorflow as tf
import matplotlib
import glob
import pandas as pd

# Retire les caractères spéciaux, les majuscules, les stop word et tokenise (stemming)
def clean_str(string,stop_word=False,token=False):
    stop = set(stopwords.words('english'))
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if stop_word:
        if not(string in stop):
            if token:
                stemmer = PorterStemmer()
                rep=stemmer.stem(string)
                return rep.strip().lower()
            else:
                return string.strip().lower()
    else:
        if token:
            stemmer = PorterStemmer()
            rep=stemmer.stem(string)
            return rep.strip().lower()
        else:
            return string.strip().lower()

# Charge les phrases et les labels
def load_data_and_labels(positive_data_file, negative_data_file,stop_word=False,token=False):
    # Chargement
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # nettoyage
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent,stop_word,token) for sent in x_text]
    # labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

#Charge Word2Vec
def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # Charchement des vecteurs Word2Vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("Erreur, fichier corrumpu")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("Erreur, fichier corrumpu")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors
