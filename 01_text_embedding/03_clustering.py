import os
import openai
from openai import OpenAI
from sklearn.cluster import KMeans
from utils import cosine_similarity

# initialize openai
os.environ['OPENAI_API_KEY']= ""
openai.api_key = os.environ["OPENAI_API_KEY"]

#
# embedding 
#

politics = ["What are the key policies of the main political parties in the upcoming election?",
            "Who do you vote for the next presedent?",
            "I love the current Democratic Party.",
            "What is your opinion on the president's current political move?",
            "I love politics. Don't you?"]

ml = ["How does supervised learning differ from unsupervised learning in machine learning models?",
      "What are the ethical considerations of using machine learning in predictive policing?",
    "How do neural networks mimic the human brain in processing data and recognizing patterns?",
    "What are some examples of natural language processing?",
    "Can you describe how machine learning is being utilized in personalized medicine and healthcare?"]

def create_embeddings(txt_list):
    client = OpenAI()

    response = client.embeddings.create(
        input=txt_list,
        model="text-embedding-3-small",
        )
    responses = [r.embedding for r in response.data]
    return responses

embeddings = politics + ml
emb = create_embeddings(embeddings)

#
# clustering
#

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(emb)

print(f"각 문장 cluster: {clusters}")

input_sentence = "I would like to have a talk about politics."
sent_emb = create_embeddings([input_sentence])

print(f"politics: {kmeans.predict(sent_emb)}")

input_sentence = "Tell me about machine learning."
sent_emb = create_embeddings([input_sentence])

print(f"machine learning: {kmeans.predict(sent_emb)}")

#
# Similarity Search
#

politics_emb = create_embeddings(politics)
ml_emb = create_embeddings(ml)

def route_selection(emb_list, query_emb, threshold=0.5):
    cos_sim = [cosine_similarity(i, query_emb) for i in emb_list]

    threshold_filtered = [i for i in cos_sim if i > threshold]

    if len(threshold_filtered) > 0:
        return True
    else:
        return False

input_sentence = "I would like to have a talk about politics."
sent_emb = create_embeddings([input_sentence])

print(f"{route_selection(politics_emb, sent_emb[0])} for politics, {route_selection(ml_emb, sent_emb[0])} for machine learning")

input_sentence = "How is the weather today?"
sent_emb = create_embeddings([input_sentence])

print(f"{route_selection(politics_emb, sent_emb[0])} for politics, {route_selection(ml_emb, sent_emb[0])} for machine learning")

input_sentence = "What is the best way to learn machine learning?"
sent_emb = create_embeddings([input_sentence])

print(f"{route_selection(politics_emb, sent_emb[0])} for politics, {route_selection(ml_emb, sent_emb[0], threshold=0.4)} for machine learning")