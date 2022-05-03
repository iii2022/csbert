import pickle
import pandas as pd
from tqdm import tqdm

with open("conceptnet.hownet/HasSubevent.pn.model.pkl","rb") as f:
    data = pickle.load(f)

relation = "/r/HasSubevent"

df = pd.read_csv("HasSubevent_pn.csv")
dct = {}
for i, row in tqdm(df.iterrows()):
    id_ = row["statement_id"]
    dct[id_] = row.to_dict()

for l in data:
    # print(l)
    ##### 去掉 head tail相同
    # print(l.keys())
    arr = l["conceptnet_triple"].strip().split("####")
    # print(arr)
    # if row["subject"].strip() == row["object"].strip():
    #     continue
    if arr[0].strip() == arr[-1].strip():
        continue

    # csv_row = dct[l["statement_id"]]
    # 负样本没有写predicate,直接写死
    # print(l["statement_id"],relation)
    # l["conceptnet_triple"] = "{}####{}####{}".format(csv_row["subject"], csv_row["predicate"], csv_row["object"])
    l["conceptnet_triple"] = "{}####{}####{}".format(arr[0].strip(), relation, arr[-1].strip())

with open("conceptnet.hownet/HasSubevent.pn.model.fix.pkl","wb") as f:
    pickle.dump(data,f)