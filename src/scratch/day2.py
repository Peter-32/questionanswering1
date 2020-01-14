#!/usr/bin/env python
# coding: utf-8

import nltk
nltk.download('punkt')
import torch
import argparse
import wikipedia
import numpy as np
from models import InferSent
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Next:

PARSER = argparse.ArgumentParser(description='Ask a question')
# Primary ETL parameters
PARSER.add_argument('--question', metavar='string', required=True,
                    help="The question you want answered")
ARGS = PARSER.parse_args()
question = ARGS.question

matches = wikipedia.search(question)
matches

# Next:

[match for match in matches]

# Next:
infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1})
infersent.load_state_dict(torch.load('/Users/petermyers/Desktop/Other/data/InferSent/encoder/infersent1.pkl'))
infersent.set_w2v_path('/Users/petermyers/Desktop/Other/data/GloVe/glove.840B.300d.txt')

# My sentences
sentences = [question] + matches
infersent.build_vocab(sentences, tokenize=True)
embeddings = infersent.encode(sentences, tokenize=True)

# Next:

embeddings = np.array(embeddings)
embeddings = StandardScaler().fit_transform(embeddings)
embeddings = MinMaxScaler().fit_transform(embeddings)
distances = pdist(embeddings, metric='euclidean')
sentence_similarity_matrix = squareform(distances)

# Next:

sentence_similarity_matrix[0]

# Next:

best_match = np.argmin(sentence_similarity_matrix[0][1:])

# Next:

# Find which sentence is most similar to the question
content_on_page = wikipedia.page(matches[best_match]).content

# Next:

import re
from nltk.tokenize import word_tokenize
wnl = nltk.WordNetLemmatizer()
# Clean Sentences
doc = content_on_page
doc = doc.lower()
sents = nltk.sent_tokenize(doc)
processed_sents = []
for sent in sents:
    words = word_tokenize(sent)
    words = [re.sub(r'[^A-Za-z_\s]', '', w) for w in words]
    words = [wnl.lemmatize(w) for w in words if w.strip() != '']
    processed_sent = " ".join(words)
    processed_sents.append(processed_sent)

# Next:

# My sentences
sentences = [question] + processed_sents
infersent.build_vocab(sentences, tokenize=True)
embeddings = infersent.encode(sentences, tokenize=True)

# Next:

embeddings = np.array(embeddings)
embeddings = StandardScaler().fit_transform(embeddings)
embeddings = MinMaxScaler().fit_transform(embeddings)
distances = pdist(embeddings, metric='euclidean')
sentence_similarity_matrix = squareform(distances)

# Next:

best_matches = list(np.argsort(sentence_similarity_matrix[0][1:]))
best_matches

# Next:

for best_match in best_matches[0:5]:
    print(processed_sents[best_match], "\n")
