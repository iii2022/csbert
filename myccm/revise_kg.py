import json, os, pickle
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

def ifrabbish(row):
    if row[5] == "revise":
        if float(row[2]) < 0.5:
            return True
    return False

def build_my_dict():
    df = pd.read_csv("common/REWEIGHT-master/conceptnet/myreweight-softmax.csv", sep="\t")
    # df = pd.read_csv("graph_light.csv.bz2", sep="\t")
    df.fillna("", inplace = True)
    df.columns = ['subject','object','score','source', "predicate", "flag"]
    df2 = df.loc[df["flag"] == "revise"]
    print(df2.shape)
    dct = {}
    for i, row in tqdm(df2.iterrows()):
        if ifrabbish(row):
            sub = convert(get_subj(row))
            obj = convert(get_obj(row))
            pre = convert(get_predicate(row))
            id_ = "{}###{}###{}".format(sub.lower(),pre.lower(),obj.lower())
            dct[id_] = [sub, pre, obj]
    return dct

def mask(s):
    return "UNK, UNK, UNK"

def toid(s):
    arr = s.split(", ")
    assert(len(arr) == 3)
    id_ = "{}###{}###{}".format(arr[0].lower(),arr[1].lower(),arr[2].lower())
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
    print(len(tmp))
    with open("data/resource.txt","r") as f:
        kg = json.load(f)
    
    c = 0
    new_triples = []
    for triple in kg["csk_triples"]:
        # print(triple)
        id_ = toid(triple)
        # print(id_)
        if id_ in tmp:
            c += 1
            # print(id_,mask(triple), triple)
            new_triples.append(mask(triple))
        else:
            new_triples.append(triple)
    print("eliminate", c)

    kg["csk_triples"] = new_triples

    d = 0
    csk_tmp = {}
    for triple, tid in kg["dict_csk_triples"].items():
        # print(triple)
        id_ = toid(triple)
        # print(id_)
        if id_ in tmp:
            # del kg[triple]
            d += 1
            continue
        else:
            # new_triples.append(triple)
            csk_tmp[triple] = tid
    kg["dict_csk_triples"] = csk_tmp
    print("delete", d)

    with open("resource.txt","w") as f:
        json.dump(kg, f)



    



    



