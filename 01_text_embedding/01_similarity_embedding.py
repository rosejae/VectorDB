#
# Word2Vec
#

from gensim.models import Word2Vec
import re

sentences = ["Homer Simpson forgot his lunch at home, so he had to buy a burger on his way to work.",
    "Marge was busy knitting a new sweater for Bart's upcoming school play.",
    "Lisa Simpson played a beautiful saxophone solo at the school concert.",
    "Mr. Burns secretly plotted another scheme from his office at the Springfield Nuclear Power Plant.",
    "Ned Flanders offered to help Homer fix the fence between their houses.",
    "Bart Simpson tried a new prank at school, but it didn't go as planned.",
    "Milhouse and Bart spent the afternoon playing video games and forgot to do their homework.",
    "Maggie Simpson's adorable giggle filled the room as she played with her toys.",
    "Apu had a busy day at the Kwik-E-Mart, dealing with a rush of customers.",
    "Krusty the Clown decided to change his show a bit to attract a new audience."]

sentences = [re.sub(r"[.',]", "", sentence).lower().split(" ") for sentence in sentences]

skip_gram = Word2Vec(sentences, vector_size=300, min_count=1, window=5, sg=1)
# print(f"homer 의 vector representation : \n{skip_gram.wv.get_vector(skip_gram.wv.key_to_index['homer'])}")
print(f'similar words to homer: {skip_gram.wv.most_similar("homer")}')

#
# Calculation of Similarity
#

homer_vector = skip_gram.wv.get_vector(skip_gram.wv.key_to_index['homer'])
marge_vector = skip_gram.wv.get_vector(skip_gram.wv.key_to_index['marge'])

import numpy as np
from numpy.linalg import norm

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    dot_product = np.dot(vector_a, vector_b)
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

print(f'cosin_similarity: {cosine_similarity(homer_vector, marge_vector)}')