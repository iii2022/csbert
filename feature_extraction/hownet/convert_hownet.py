import pickle
from commons.util import *

def split_triple(triples):
    res = []
    for t in triples:
        arr = t.strip().split("    ")
        res.append(arr)
    return res

if __name__ == "__main__":
	
    with open("1HowNet.json","r") as f:
        data = json.load(f)
    
    dct = {}
    for k,v in data.items():
        triples = v["DEF"].keys()
        arr = split_triple(triples)
        label = v["label"]
        id_ = k
        dct[id_] = {"triples":arr, "label":label}
    # print(dct)
    with open("convert_hownet.pkl","wb") as f:
        pickle.dump(dct, f)

