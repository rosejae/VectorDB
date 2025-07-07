from typing import Optional
import nltk
import re
import pandas as pd
import spacy

nltk.download('punkt')
nltk.download('stopwords')

#
# load data
#

df = pd.read_csv('simpsons_dataset.csv')
print(f'df.shape: \n{df.shape}')
print(f'null values: \n{df.isnull().sum()}')

#
# preprocessing
#

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

cleaner = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])
txt = [cleaning(doc) for doc in nlp.pipe(cleaner, batch_size=5000)]

print(f'original sentence: \n{df.loc[0, "spoken_words"]}')
print(f'outcome: \n{txt[0]}')

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
print(f'df_clean.shape: \n{df_clean.shape}')

sentences = [s.split(' ') for s in df_clean['clean']]

#
# train
#

from gensim.models import Word2Vec

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007)

w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=100)

#
# inference
#

print(f'similar words to homer: \n{w2v_model.wv.most_similar(positive=["homer"])}\n')
print(f'similar words to bart: \n{w2v_model.wv.most_similar(positive=["bart"])}\n')

print(f'most similar: \n{w2v_model.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3)}\n')
print(f'doesnt match: \n{w2v_model.wv.doesnt_match(['bart', 'homer', 'marge'])}\n')

#
# sentence embedding
#

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
model = BertModel.from_pretrained('bert-base-uncased')

sentence1 = "I deposited money at the bank."
sentence2 = "The ducks swam to the river bank."

encoded_input1 = tokenizer(sentence1, return_tensors='pt') 
encoded_input2 = tokenizer(sentence2, return_tensors='pt')

with torch.no_grad():
    output1 = model(**encoded_input1)
    output2 = model(**encoded_input2)

bank_embedding_sentence1 = output1.last_hidden_state[0, 5, :]
bank_embedding_sentence2 = output2.last_hidden_state[0, 5, :]

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    dot_product = np.dot(vector_a, vector_b)
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

similarity = cosine_similarity(bank_embedding_sentence1, bank_embedding_sentence2)
# print("Embedding for 'bank' in sentence 1:", bank_embedding_sentence1)
# print("Embedding for 'bank' in sentence 2:", bank_embedding_sentence2)
print("Cosine similarity between the two embeddings:", similarity)

