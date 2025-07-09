#
# Simpson dataset
#

import pandas as pd
from collections import Counter

df = pd.read_csv(r'.\simpsons_dataset.csv')

counts = Counter(df['raw_character_text'])
counts.most_common()

#
# Quora dataset
#

from datasets import load_dataset
import pandas as pd

dataset = load_dataset("quora", trust_remote_code=True)
raw_df = dataset["train"].to_pandas()

raw_df = raw_df.loc[raw_df['is_duplicate']==True].reset_index(drop=True)

raw_df["q1"] = raw_df["questions"].apply(lambda q: q["text"][0])
raw_df["q2"] = raw_df["questions"].apply(lambda q: q["text"][1])
raw_df["id1"] = raw_df["questions"].apply(lambda q: q["id"][0])
raw_df["id2"] = raw_df["questions"].apply(lambda q: q["id"][1])

q1_to_q2 = raw_df.copy().rename(columns={"q1": "text", "id1": "id", "id2": "dq_id"}).drop(columns=["questions", "q2"])
q2_to_q1 = raw_df.copy().rename(columns={"q2": "text", "id2": "id", "id1": "dq_id"}).drop(columns=["questions", "q1"])

flat_df = pd.concat([q1_to_q2, q2_to_q1])
flat_df = flat_df.sort_values(by=['id']).reset_index(drop=True)

# use only the first 15,000 entries from the dataset
flat_df = flat_df.loc[((flat_df['id'] <= 15000) & (flat_df['dq_id'] <= 15000))]

# duplicated_questions made of list
# final format
df = flat_df.drop_duplicates("id")
df.loc[:, "duplicated_questions"] = df["id"].apply(lambda qid: flat_df[flat_df["id"] == qid]["dq_id"].tolist())
df = df.drop(columns=["dq_id", "is_duplicate"])
df.loc[:, 'length'] = [len(x) for x in df['duplicated_questions']]
df.to_csv("quora_dataset.csv")

#
# abcnews dataset
#

import pandas as pd
df = pd.read_csv(r".\abcnews.csv")

# use only the data from 20/01/01 ~ 20/02/01
# df.publish_date.max(), df.publish_date.min()
news_2020 = df.loc[(df['publish_date']>=20200101) & (df['publish_date']<20200201)].reset_index(drop=True)
news_2020.to_csv("abcnews_2020.csv", index=False)

#
# Resume dataset
#

import pandas as pd
resume = pd.read_csv(r".\Resume.csv")
# resume.Category.unique()