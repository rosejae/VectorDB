import os
import random
import json

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from utils import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
import cohere
import openai
from openai import OpenAI

import warnings
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

now_path = './'
with open(f'{now_path}/metadata/setting.json', 'r', encoding='utf-8') as file:
    config_dict = json.load(file)

# initialize openai
os.environ["OPENAI_API_KEY"] = config_dict['OPENAI_KEY']
openai.api_key = os.environ["OPENAI_API_KEY"]

# initialize cohere
os.environ["CO_API_KEY"] = config_dict['COHERE_KEY']
co = cohere.Client()

#
# load data
#

df = pd.read_csv("quora_dataset.csv")

#
# measurement of  similarity using openai or cohere
#

def create_embeddings(txt_list, provider='openai'):
    if provider=='openai':
        client = OpenAI()

        response = client.embeddings.create(
            input=txt_list,
            model="text-embedding-3-small",
            )
        responses = [r.embedding for r in response.data]
        return responses
    
    elif provider=='cohere':
        doc_embeds = co.embed(
            txt_list,
            input_type="search_document",
            model="embed-english-v3.0",
            )
        return doc_embeds.embeddings
    else:
        assert False, "Double check provider name"

emb1 = create_embeddings(df.loc[2, 'text'])
emb2 = create_embeddings(df.loc[3, 'text'])

print(f"Cosine 유사도 : {cosine_similarity(emb1[0], emb2[0])}.\n사용된 문장 : \n{text1}\n{text2}")

#
# embedding vector dataset from raw dataset
#

# openai embedding model
openai_emb = create_embeddings(df.text.tolist(), provider='openai')
df['openai_emb'] = openai_emb

# cohere embedding model
cohere_emb = create_embeddings(df.text.tolist(), 'cohere')
df['cohere_emb'] = cohere_emb

# e5 embedding models
device = "cuda" 
model_id = "intfloat/e5-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

def create_e5_emb(docs, model):
    docs = [f"query: {d}" for d in docs]
    tokens = tokenizer(docs, padding=True, max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens)
        last_hidden = out.last_hidden_state.masked_fill( # from last hidden state
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        # average out embeddings per token (non-padding)
        doc_embeds = last_hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
    return doc_embeds.cpu().numpy()

data = df.text.tolist()
batch_size = 128

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    data_batch = data[i:i_end]
    # embed current batch
    embed_batch = create_e5_emb(data_batch, model)
    if i == 0:
        emb3 = embed_batch.copy()
    else:
        emb3 = np.concatenate([emb3, embed_batch.copy()])

# numpy to list
emb3 = [list(e) for e in emb3]
df['e5_emb'] = emb3

# df.to_csv("quora_dataset_emb.csv", index=False)

#
# test dataset
#

df_new = pd.read_csv("quora_dataset_emb.csv")

# str -> list 형태로 변환
df_new['openai_emb'] = df_new['openai_emb'].apply(json.loads)
df_new['cohere_emb'] = df_new['cohere_emb'].apply(json.loads)
df_new['e5_emb'] = df_new['e5_emb'].apply(json.loads)
df_new['duplicated_questions'] = df_new['duplicated_questions'].apply(json.loads)

test_query = random.choices(df_new.id, k=1000)
test = df_new.loc[df_new.id.isin(test_query)]

from sklearn.metrics.pairwise import cosine_similarity

def search_top_k(search_df, search_df_column, id, topk):
    """
    search_df : search를 할 대상 dataframe
    search_df_column : search를 위해 사용될 embedding column name (out of openai, cohere, e5)
    id : test query id
    topk : 유사도 기반으로 top-k개 선별
    """
    query = search_df.loc[search_df['id']==id, search_df_column].values[0]
    query_reshaped = np.array(query).reshape(1, -1)
    
    search_df = search_df.loc[search_df['id']!=id]
    # cosine similarity in batch
    similarities = cosine_similarity(query_reshaped, np.vstack(search_df[search_df_column].values)).flatten()
    
    search_df['similarity'] = similarities
    
    # Get top-k indices
    # hence we sort the topk indices again to ensure they are truly the top-k
    topk_indices = np.argpartition(similarities, -topk)[-topk:]
    topk_indices_sorted = topk_indices[np.argsort(-similarities[topk_indices])]
    
    # Retrieve the top-k results
    search_result = search_df.iloc[topk_indices_sorted]
    
    return search_result

query_results_openai = {k:search_top_k(df_new, 'openai_emb', k, 5) for k in test.id}
query_results_cohere = {k:search_top_k(df_new, 'cohere_emb', k, 5) for k in test.id}
query_results_e5 = {k:search_top_k(df_new, 'e5_emb', k, 5) for k in test.id}

#
# evaluation
#

def score_accuracy(full_df, tmp_df, test_id):
    """
    각 테스트 질문과 유사하다고 판단된 질문들 중, 실제 duplicated_questions에 들어있는 질문들을 count
    full_df: 우리의 database
    tmp_df: retrieve된 결과
    test_id
    """
    duplicated_questions = full_df.loc[full_df['id'] == test_id, 'duplicated_questions'].values[0]

    # 본인 ID는 제외
    filtered_df = tmp_df[tmp_df['id'] != test_id]
    # 현재 retrieve 해온 ID들이, 테스트 질문 내에 들어있는 아이디들인지 count
    match_count = filtered_df['id'].isin(duplicated_questions).sum()

    # Calculate the accuracy in terms of percentage
    if filtered_df.shape[0] < len(duplicated_questions):
        percentage = (match_count / filtered_df.shape[0])
    else:
        percentage = (match_count / len(duplicated_questions))
    return percentage

accuracy_openai = [score_accuracy(df_new, query_results_openai[i], i) for i in query_results_openai.keys()]
accuracy_cohere = [score_accuracy(df_new, query_results_cohere[i], i) for i in query_results_cohere.keys()]
accuracy_e5 = [score_accuracy(df_new, query_results_e5[i], i) for i in query_results_e5.keys()]

print(f"accuracy_openai: {accuracy_openai}")
print(f"accuracy_cohere: {accuracy_cohere}")
print(f"accuracy_e5: {accuracy_e5}")

indices = [index for index, value in enumerate(accuracy_openai) if value <= 0.5]

# list(query_results_openai.keys())[81]
# test.loc[test['id']==1178]
# query_results_openai[1178]