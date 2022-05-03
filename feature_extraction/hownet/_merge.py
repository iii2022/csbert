from commons.util import *
from tqdm import tqdm
import os, pickle

if __name__ == "__main__":

    # triples
    names_triple = list_files("conceptnet.triples/triples/", "*.csv")
    triple_pkl = "aligment/triples.pkl"
    name_aligment = "aligment/match_result.txt"
    aligment_pkl = "aligment/aligment.pkl"
    triple_weight = "conceptnet.weight/contextall.csv"
    triple_weight_pkl = "conceptnet.weight/weight.pkl"

    print("building statemnets dict")
    # build statement_id dict
    dct = {}
    if not os.path.isfile(triple_pkl):

        for name in names_triple:
            df = pd.read_csv(name)
            df.fillna("", inplace = True)
            for i, row in tqdm(df.iterrows()):
                id_ = row["statement_id"]
                dct[id_] = row.to_dict()
        print("dumps to pkl")
        with open(triple_pkl, 'wb') as fpkl:
            pickle.dump(dct, fpkl)
    else:
        print("loading triple pickle")
        with open(triple_pkl, 'rb') as fpkl:
            dct = pickle.load(fpkl)

    print("start aligment pkl")
    print("read weight")
    weight_dct = {}
    if not os.path.isfile(triple_weight_pkl):
        df = pd.read_csv(triple_weight)
        df.fillna("", inplace = True)
        for i, row in tqdm(df.iterrows()):
            id_ = row["statement_id"]
            weight_dct[id_] = row.to_dict()
        print("dumps to pkl")
        with open(triple_weight_pkl, 'wb') as fpkl:
            pickle.dump(weight_dct, fpkl)
    else:
        print("loading weight pickle")
        with open(triple_weight_pkl, 'rb') as fpkl:
            weight_dct = pickle.load(fpkl)

    aligment = []
    with open(name_aligment,"r") as f:
        for d in tqdm(parse_alignment(f)):
            # print(d)
            id_ = d["statement_id"]
            print(id_)
            # print(id_ in weight_dct)
            if id_ in dct:
                row_ = dct[id_]
                d.update(row_)
            if id_ in weight_dct:
                weight_row_ = weight_dct[id_]
                d.update(weight_row_)
            # print(d)
            aligment.append(d)

    with open(aligment_pkl, 'wb') as fpkl:
        pickle.dump(aligment, fpkl)
    
