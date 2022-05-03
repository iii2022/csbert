import pickle
import numpy as np

with open("/ldata/name/common/PartOf_pandn_instance_noxiaoqi/all/test.pkl","rb") as f:
    d = pickle.load(f)

tmp = []
for l in d:
    # print(l)
    # print(l["statement_id"])
    if "nodeID" not in str(l["statement_id"]):
        continue
    paths = l["hownet_paths"]
    paths.sort(key=lambda s: len(s))
    for p in paths[:10]:
        # print(len(p))
        # print(p)
        tmp.append(len(p))
print("mean", np.mean(tmp))
print("min", np.min(tmp))
print("max", np.max(tmp))