import pandas as pd

pos = pd.read_csv("HasContext_match_result.csv")
neg = pd.read_csv("HasContext_neg_match_result.csv")

df = pd.concat([pos,neg],axis=0)

df.to_csv("HasContext_pn.csv")