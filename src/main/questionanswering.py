# Initialize
#### Imports
import os
import re
import nltk
import torch
import argparse
import wikipedia
import numpy as np
from models import InferSent
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#### Parameters
PARSER = argparse.ArgumentParser(description='Ask a question')
PARSER.add_argument('--question', metavar='string', required=True, help="The question you want answered")
ARGS = PARSER.parse_args()
question = ARGS.question
sentences = [question]
#### Load Facebook's InferSent
infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1})
infersent.load_state_dict(torch.load('/Users/petermyers/Desktop/Other/data/InferSent/encoder/infersent1.pkl'))
infersent.set_w2v_path('/Users/petermyers/Desktop/Other/data/GloVe/glove.840B.300d.txt')

# Extract the most relevant Wikipedia page
#### Wikipedia recommends 10 pages
sentences = sentences + wikipedia.search(question)
#### Convert sentences to numbers
infersent.build_vocab(sentences, tokenize=True)
embeddings = infersent.encode(sentences, tokenize=True, verbose=False)
#### Choose the most relevant page
embeddings = np.array(embeddings)
distances = pdist(embeddings, metric='euclidean')
sentence_similarity_matrix = squareform(distances)
best_matches = np.argsort(sentence_similarity_matrix[0][1:])
#### Extract the content on the page
for best_match in best_matches:
    try:
        content_on_the_page = wikipedia.page(matches[best_match]).content
        break
    except:
        pass

# Find and print the most relevant sentences
#### Split the content into sentences
sents = nltk.sent_tokenize(content_on_the_page)
#### Convert sentences into numbers
sentences = [question] + sents
infersent.build_vocab(sentences, tokenize=True)
embeddings = infersent.encode(sentences, tokenize=True, verbose=False)
#### Choose the most relevant sentences
embeddings = np.array(embeddings)
distances = pdist(embeddings, metric='euclidean')
sentence_similarity_matrix = squareform(distances)
most_relevant_sentences = list(np.argsort(sentence_similarity_matrix[0][1:]))
#### Print the most relevant sentences
print("\n")
found=0
for sentence in most_relevant_sentences:
    if found >= 5:
        break
    if len(sents[sentence]) > 60:
        found+=1
        print(sents[sentence], "\n")
