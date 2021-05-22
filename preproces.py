import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hiring (1).csv")
nums = {"experience": {"two": 2, "three": 3, "eleven": 11, "seven": 7, "ten": 10, "five": 5}}
df = df.replace(nums)
df['experience'].fillna(0, inplace=True)
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(), inplace=True)
x = df.drop('salary($)', axis=1)
scaler = StandardScaler()
df_sc = scaler.fit(x)
def prepro(x):
    x_sc = scaler.transform(x)
    return x_sc

print("run__")
