import pandas
import pandas as pd

from collections import Counter


# majority voting
def vote(x):
    return Counter(x).most_common()[0]


l = range(1, 15+1)
df = pd.concat([pd.read_csv(f"{x}.csv", dtype={'Filename': str}).sort_values('Filename').reset_index() for x in l], axis=1)



df["counter"] = df.ClassId.apply(lambda x: Counter(x), axis=1)
df["vote"] = df.counter.apply(vote)

df_out = pd.read_csv("gtsrb_kaggle.csv", dtype={'Filename': str}).sort_values('Filename').reset_index()
df_out["ClassId"] = df.vote.apply(lambda x: x[0])
df_out[["Filename", "ClassId"]].to_csv("big-densenet169-xaug-vote3.csv", index=None)


