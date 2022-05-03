import numpy as np
import pickle
from scipy.special import softmax

pickle_path = "/ldata/name/common/PartOf_pandn_instance_noxiaoqi/all/test.pkl"
score_path = "predictions/partof-onlytriple/predict_scores.npy"
output = "predictions-softmax/partof-onlytriple/reweight_relation.pkl"
with open(pickle_path,"rb") as f:
    data = pickle.load(f)
score = np.load(score_path, allow_pickle=True)
# print(score[0])
logits_, labels, ids = [], [], []
for logits, label, id_ in score:
    logits_.append(logits)
    labels.append(label)
    ids.append(id_)

logits_vec = np.concatenate(logits_, axis=0)
############## softmax
logits_vec = softmax(logits_vec, axis=-1)

labels_vec = np.concatenate(labels, axis=0)
ids_vec = np.concatenate(ids, axis=0)

print(logits_vec.shape)
logits_sort = np.sort(logits_vec[:,1])
descend_value = logits_sort[::-1]
logits_ind_sort = np.argsort(logits_vec[:,1])
descend_ind = logits_ind_sort[::-1]
# print(descend_sort)
# norm_descend = (descend_value - min(descend_value)) / (max(descend_value)-min(descend_value))
norm_descend = descend_value
print(min(descend_value), max(descend_value))
res = []
tmp = {}
for j,(i,v) in enumerate(zip(descend_ind, norm_descend)):
    if "nodeID" in str(data[i]["statement_id"]):
        res.append(data[i])
        tmp[str(data[i]["conceptnet_triple"])] = v
    # print(i,v)
    # print(data[i]["statement_id"])
# for s in res[-10:]:
#     print(s["conceptnet_triple"])

np.save(output, tmp)