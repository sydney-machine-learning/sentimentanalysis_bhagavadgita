# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:34:49 2021

@author: Sweta
"""
import re
import nltk
from nltk.corpus import stopwords
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

fileName = "../textFiles/Gita-According-to-Gandhi"
with open(fileName + "_refined2.txt", 'r', encoding="utf-8") as f:
    data = f.read()
    f.close()
    
    
data = data.replace("\n", " ")
data = re.sub(r'\s+', " ", data)
data = data.lower()
data = re.sub(r'\s+', " ", data)
sentences = re.split('[0-9]+\\.', data)[1:]
sentences = [s.strip() for s in sentences]

vectorizer = Vectorizer()
vectorizer.bert(sentences[:345])
vectors_bert = vectorizer.vectors

dist_1 = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])
dist_2 = spatial.distance.cosine(vectors_bert[0], vectors_bert[2])
