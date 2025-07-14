import json
import os
import openai
import asyncio

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI

load_dotenv()

#
# Extracting data using GPT
#

client = openai.OpenAI()

df = pd.read_csv(r"./Resume.csv")
df = df.loc[df.Category.isin(['CHEF', 'FITNESS'])].reset_index(drop=True)

prompt = """Given the following resume text, 
extract and categorize the information into the specified categories: skills, work experience in years, and summary of each project. 
Please provide the extracted information in a dictionary format with the keys as 'skills', 'work experience (years)' and 'summary.

Instructions:

    Skills: Identify and list all professional skills mentioned in the resume. Each element should be a word such as 'Python' or 'CSS'
    Work Experience (years): Total years of experience. It should be a number such as '7' or '10'. Leave it empty if there are no related information.
    Summary : For each career should be one summarized in one sentence. 
              Each sentence should be in a format of 'Worked as <job_title> from <start_date> to <end_date>, doing <work description> and accomplishing <accomplishments>'.
              Put in 'empty' for each blank if there are on relevant information.
    
Ensure that the information is accurately extracted and categorized according to the instructions. If certain information is not available or cannot be accurately determined, please indicate so appropriately.

Resume Text:
{}
"""

async def chat_completion(input_prompt, model='gpt-4o'):
    client = AsyncOpenAI()
    
    SYSTEM_PROMPT = "You are a smart and intelligent program that understands information inside a resume, designed to output JSON"
    USER_PROMPT_1 = """Are you clear about your role?"""
    ASSISTANT_PROMPT_1 = """Sure, I'm ready to help you with your NER task. Please provide me with the necessary information to get started."""

    response = await client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content":input_prompt}
        ]
        )
    return response

async def run_async(main_prompt, information):
    tasks = [chat_completion(main_prompt.format(i)) for i in information]
    responses = await asyncio.gather(*tasks)
    return responses

batches = (df.Resume_str[i:i+5].values.tolist() for i in range(0, len(df.Resume_str), 5))
outputs = list()

for batch in tqdm(batches):
    try:
        output = await run_async(prompt, batch)
        outputs.extend(output)
    except openai.RateLimitError:
        print("Rate limit hit, waiting 2 seconds...")
        await asyncio.sleep(2)

extracted_info = [i.choices[0].message.content for i in outputs]
extracted_info = [json.loads(i) for i in extracted_info]

for i, info in enumerate(extracted_info):
    extracted_info[i]['ID'] = str(df.loc[i, 'ID'])
    extracted_info[i]['title'] = df.loc[i, 'title']
    
with open(r".\resume\resume_info_extracted.json", 'w') as file:
    json.dump(extracted_info, file)

#
# Embedding feature using OpenAI
#

from utils import create_embeddings

with open(r".\resume\resume_info_extracted.json", 'r') as file:
    data = json.load(file)

emb_data = list()

for d in tqdm(data):
    emb_d = dict()
    for k, v in d.items():
        if k in ['skills', 'summary', 'title']:
            emb_ = create_embeddings(v) # list를 한 번에 embedding화
            emb_d[k] = emb_
        elif k in ['work experience (years)', 'ID']:
            emb_d[k] = v
        else:
            assert False, "Incorrect key"
    emb_data.append(emb_d)

with open(r".\resume\resume_info_extracted_emb.json", 'w') as file:
    json.dump(emb_data, file)

#
# Search based on skills
#

with open(r".\resume\resume_info_extracted.json", 'r') as file:
    data = json.load(file)

with open(r".\resume\resume_info_extracted_emb.json", 'r') as file:
    emb_data = json.load(file)

df = pd.DataFrame(data)
emb_df = pd.DataFrame(emb_data)

input_dict = {'skills':['Flexibility Training', 'Nutrition', 'Anatomy', 'Strength Training'],
              'summary':"Extensive experience in designing and implementing personalized training programs for muscle growth, with a proven track record of helping clients achieve their fitness goals"}

def batch_cosine_similarity(list1, list2, threshold):
    # sklearn의 cosine_similarity 함수를 사용하여 코사인 유사도 계산
    similarities = cosine_similarity(list1, list2)
    columns_over_threshold = (similarities > threshold).any(axis=0)
    
    count = columns_over_threshold.sum() # list2를 기준으로 한 개라도 threshold를 넘는 값이 있으면 +1
    column_indices = np.where(columns_over_threshold)[0]

    return column_indices, count

def candidate_search(input_list, nested_lists, top_k, search_type='skill', threshold=0.5):
    """
    score : 0-1 사이의 값. 높을 수록 더 많은 match. Match의 max는 nested_list의 개수와 동일
    현재 input으로 제공된 embedding 값과, nested_lists에 있는 element들의 embedding 값들의 cosine similarity를 계산
    """
    if search_type in ['experience', 'skill']:
        pass
    else:
        assert False, "Unsupported search type"

    scores = list()
    
    for i, nested_list in enumerate(nested_lists):
        # input_list와 nested_lists를 대상으로 cosine similarity를 계산, 각 element 별로 cos_sim이 threshold를 넘는 값들만 가져옴
        _, common_elements_count = batch_cosine_similarity(input_list, nested_list, threshold)
        # print(common_elements_count)
        # 정규화를 위해 nested_list의 길이 계산
        possible_matches = len(nested_list)
        # 점수 계산 (common_elements_count / possible_matches)
        score = common_elements_count / possible_matches if possible_matches > 0 else 0
        scores.append((i, score))
    
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return top_scores

db = emb_df['skills'].values.tolist()
input = create_embeddings(input_dict['skills'])

skill_based_findings = candidate_search(input, db, 10, threshold=0.5)
print(f'skill_based_findings: {skill_based_findings}')

#
# Search based on experience
#

summary_db = emb_df['summary'].values.tolist()
input_summary = create_embeddings(input_dict['summary'])

summary_based_findings = candidate_search(input_summary, summary_db, 10, 'experience')
print(f'summary_based_findings: {summary_based_findings}')

#
# Extra work for service (step 1)
#

job_req = "I want to grow muscle mass considering nutrient intake as well as various muscle training drills."

job_search_queries = ["Improving physical fitness through a combination of general physical education activities, balanced exercise routines, and nutritional awareness.",
                      "Enhancing overall health with a mix of diverse physical education exercises, targeted workouts, and mindful eating habits.",
                      "Increasing muscle volume by integrating nutritional strategies with multifaceted workout routines.",
                      "Building muscle density by focusing on nutrient-rich diets and comprehensive resistance training programs.",
                      "Streamlining trainer scheduling and client management to optimize the efficiency and effectiveness of a fitness facility.",
                      "Implementing cutting-edge fitness technology and equipment maintenance protocols to ensure a state-of-the-art workout environment.",
                      "Developing comprehensive staff training programs to elevate the expertise and service quality of personal trainers and fitness instructors.",
                      "Enforcing health and safety standards to provide a secure and hygienic environment for members and staff alike.",
                      "Cultivating a community-focused atmosphere through member engagement initiatives and personalized fitness guidance to enhance client retention and satisfaction."]

job_search_query_embs = create_embeddings(job_search_queries)
input_emb = create_embeddings(job_req)

def route_selection(query_emb, emb_list, threshold=0.5):
    cos_sim = cosine_similarity(query_emb, emb_list)
    threshold_check = cos_sim > threshold

    if threshold_check.sum() > 0:
        return True
    else:
        return False

route_selection(input_emb, job_search_query_embs)

#
# Extra work for service (step 2)
#

prompt = """
Analyze the provided task description to identify and categorize the essential qualifications and expertise required for the job. 
The analysis should focus on extracting relevant skills and summarizing the job capabilities necessary for achieving the specified goal.
Organize this information into a structured dictionary format.

Categories: Skills and Summary.

Instructions:
- Skills: Enumerate the critical skills necessary for someone to effectively fulfill the job requirements. These should be simple words such as 'Anatomy' or 'Strength Training'
- Summary: Draft a concise job description that encapsulates the professional experience and competencies needed to successfully execute the job responsibilities. 
            One example would be : "Extensive experience in designing and implementing personalized training programs for muscle growth, with a proven track record of helping clients achieve their fitness goals"

Please provide the extracted information in a dictionary format with the keys as 'skills' and 'summary'.

Task description:
{}
"""

def normal_chat_completion(input_prompt, model='gpt-4o'):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are a smart and intelligent program that understands information and provides output in JSON format'},
            {"role": "user", "content":input_prompt}
        ]
        )
    return response

output = normal_chat_completion(prompt.format(job_req))
print(f'output: {json.loads(output.choices[0].message.content)}')