import nltk
nltk.download('punkt')
import torch
from models import InferSent

# Initialize
infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1})
infersent.load_state_dict(torch.load('/Users/petermyers/Desktop/Other/data/InferSent/encoder/infersent1.pkl'))
infersent.set_w2v_path('/Users/petermyers/Desktop/Other/data/GloVe/glove.840B.300d.txt')

# My sentences
sentences = ["Hi I'm Peter", "Hi I'm Danny", "Hi I'm Ryan"]
infersent.build_vocab(sentences, tokenize=True)
embeddings = infersent.encode(sentences, tokenize=True)
print(embeddings)
infersent.visualize(sentences[0], tokenize=True)
