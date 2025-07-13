import pandas as pd
import os
import json
import openai
from openai import OpenAI
import numpy as np
from tqdm.notebook import tqdm, trange
from sklearn.cluster import KMeans
from utils import create_embeddings

# initialize openai
os.environ['OPENAI_API_KEY']= ""
openai.api_key = os.environ["OPENAI_API_KEY"]

#
# load data
#

df = pd.read_csv("abcnews_2020.csv")
df.head()

#
# clustering
#

# 비용 발생 주의
batch_size = 2000
headline_emb = list()

headline = df['headline_text'].tolist()

for i in trange(0, len(headline), batch_size):
    i_end = min(len(headline), i+batch_size)
    data_batch = headline[i:i_end]

    tmp_emb = create_embeddings(data_batch)
    headline_emb.extend(tmp_emb)

df['headline_emb'] = headline_emb

# df.to_csv("abcnews_2020_emb.csv", index=False)
# df = pd.read_csv("abcnews_2020_emb.csv")

# df['headline_emb'] = df['headline_emb'].apply(json.loads)
# type(df.loc[0, 'headline_emb'])

clusters = KMeans(n_clusters=15, random_state=0).fit_predict(df['headline_emb'].tolist())
df['cluster'] = clusters

#
# Diversity of information
#

from sklearn.metrics.pairwise import cosine_similarity

def calculate_diversity(df, column_name):
    """
    Calculates the diversity of a set of embeddings based on cosine distance.
    
    :param embeddings: NumPy array of embeddings
    :return: The average cosine distance between embeddings, higher means more diverse
    """
    # 각각의 임베딩끼리 모두 pairwise cosine similarity를 계산
    embeddings = np.vstack(df[column_name])
    cosine_sim = cosine_similarity(embeddings)
    
    # self-comparisons (diagonal elements)를 제외하고 cosine similarity 계산
    np.fill_diagonal(cosine_sim, np.nan) # 본인과의 similarity는 제외
    avg_distance = np.nanmean(cosine_sim)
    
    return cosine_sim, avg_distance

dist, avg = calculate_diversity(df, 'headline_emb')

diversity_score = {k:calculate_diversity(df.loc[df['cluster']==k], 'headline_emb')[1] for k in range(0, 15)}

#
# measurement of outliers
#

from sklearn.ensemble import IsolationForest

cluster = df.loc[df['cluster']==3]
iso_forest = IsolationForest(contamination=0.05)  # Adjust contamination as needed
anomalies = iso_forest.fit_predict(cluster['headline_emb'].tolist())

anomalous_headlines = np.array(cluster['headline_text'].tolist())[anomalies == -1]
print(f'Anomalous Headlines: {anomalous_headlines}')