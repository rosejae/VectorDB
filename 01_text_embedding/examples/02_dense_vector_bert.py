import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# 변환 함수
def create_dense_vector(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# 비교 예시 문장 입력
sentences = ["Bill ran from the giraffe toward the dolphin",
             "Bill ran from the dolphin toward the giraffe"]

# Dense Vectorizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 예시문장에 대해 덴스벡터 생성
dense_vectors = [create_dense_vector(sentence) for sentence in sentences]

# 결과 출력
for vector in dense_vectors:
    print(vector)

# 덴스벡터 기준 코사인 유사도 계산
def calculate_cosine_similarity(vector1, vector2):
    return F.cosine_similarity(vector1, vector2).item()

dense_similarity = calculate_cosine_similarity(dense_vectors[0], dense_vectors[1])

print("Cosine Similarity (Dense Vectors):", dense_similarity)