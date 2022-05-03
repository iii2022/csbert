import json, os, pickle, gzip
import pandas as pd
from tqdm import tqdm

def convert(s):
    s = s.strip()
    labels = s.split("/")
    if len(labels) == 5:
        return labels[3].replace("_", " ")
    if len(labels) < 5:
        return labels[-1].replace("_", " ")
    if len(labels) > 5:
        return labels[3].replace("_", " ")

def get_subj(row):
    return row[0].strip()

def get_obj(row):
    return row[1].strip()

def get_predicate(row):
    return row[4].strip()

def get_score(row):
    return float(row[2])

# def ifrevise(row):
#     if row[5] == "revise":
#         return True
#     return False
    #     if float(row[2]) < 0.5:
    #         return True
    # return False

def build_my_dict(p="data/cpnet/myreweight-softmax.csv"):
    df = pd.read_csv(p, sep="\t")
    # df = pd.read_csv("graph_light.csv.bz2", sep="\t")
    df.fillna("", inplace = True)
    df.columns = ['subject','object','score','source', "predicate", "flag"]
    df2 = df.loc[df["flag"] == "revise"]
    print(df2.shape)
    dct = {}
    for i, row in tqdm(df2.iterrows()):
        # if ifrevise(row):
        sub = get_subj(row)
        obj = get_obj(row)
        pre = get_predicate(row)
        id_ = "{}###{}###{}".format(sub.lower(),pre.lower(),obj.lower())
        dct[id_] = get_score(row)
    return dct

def revise(l, new_score):
    if new_score < 0.5:
        return None
    arr = l.split("\t")
    assert(len(arr) == 5)
    js = json.loads(arr[-1])
    
    js["weight"] = new_score
    js["revise-name"] = "YES"
    arr[-1] = json.dumps(js)
    new_l = "\t".join(arr)
    return new_l

def toid(s):
    arr = s.split("\t")
    assert(len(arr) == 5)
    id_ = "{}###{}###{}".format(arr[2].strip().lower(),arr[1].strip().lower(),arr[3].strip().lower())
    return id_

if __name__ == "__main__":
    tmp = None
    if not os.path.exists("myreweight.pkl"):
        tmp = build_my_dict()
        with open("myreweight.pkl", "wb") as f:
            pickle.dump(tmp, f)
    else:
        with open("myreweight.pkl","rb") as f:
            tmp = pickle.load(f)

    tmp2 = None
    if not os.path.exists("myreweight2.pkl"):
        tmp2 = build_my_dict(p="/ldata/name/common/REWEIGHT-master/conceptnet/myreweight-noxiaoqi-onlyhownet.csv")
        with open("myreweight2.pkl", "wb") as f:
            pickle.dump(tmp2, f)
    else:
        with open("myreweight2.pkl","rb") as f:
            tmp2 = pickle.load(f)

    tmp3 = None
    if not os.path.exists("myreweight3.pkl"):
        tmp3 = build_my_dict(p="/ldata/name/common/REWEIGHT-master/conceptnet/myreweight-noxiaoqi-onlytriple.csv")
        with open("myreweight3.pkl", "wb") as f:
            pickle.dump(tmp3, f)
    else:
        with open("myreweight3.pkl","rb") as f:
            tmp3 = pickle.load(f)
    # print(len(tmp))
    res = []
    # with open("data/cpnet/conceptnet-assertions-5.6.0.csv","r") as f:
    with gzip.open("data/cpnet/conceptnet-assertions-5.6.0.csv.gz", 'rb') as f:
        # kg = json.load(f)
        for l in f:
            res.append(l.decode("utf8").strip())
    c = 0
    d = 0
    new_cpnet = []
    for l in tqdm(res):
        # print(triple)
        id_ = toid(l)
        # print(id_)
        if id_ in tmp:
            c += 1
            # print(id_,mask(triple), triple)
            fc = 0
            
            ls = revise(l, tmp[id_])
            ls2 = revise(l, tmp2[id_])
            ls3 = revise(l, tmp3[id_])
            if ls is not None:
                fc += 1
            if ls2 is not None:
                fc += 1
            if ls3 is not None:
                fc += 1
            if fc >= 2:
                d += 1
                continue
            new_cpnet.append(l)
        else:
            new_cpnet.append(l)
    print("revise", c)
    print("eliminate", d)


    with open("data/cpnet/new_conceptnet3.csv","w") as f:
        f.write("\n".join(new_cpnet))
