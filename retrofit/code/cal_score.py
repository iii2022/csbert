import pickle
from sklearn.metrics import accuracy_score

def mag(l, predict, field = "conceptnet"):
    margin = l[-2][field] - l[-1][field]
    if margin >= -0.1 and margin <= 0.1:
        predict.append(0)
    if margin > 0.1:
        predict.append(1)
    if margin < -0.1:
        predict.append(2)

def gethuman(p, gold):
    with open(p,'r') as f:
        for l in f:
            if l[0]=="=":
                gold.append(0)
            if l[0]==">":
                gold.append(1)
            if l[0]=="<":
                gold.append(2)
    


if __name__ == "__main__":
    fs = ["example.AtLocation.pkl", 
    "example.CapableOf.pkl",
    "example.PartOf.pkl",
    "example.UsedFor.pkl"
    ]

    
    for name in fs:
        predict = []
        predictbert = []
        predicthownet = []
        predictberthownet = []
        gold = []
        with open(name, "rb") as f:
            d = pickle.load(f)
            for l in d:
                # print(l[-2], l[-1])
                mag(l, predict, field="conceptnet")
                mag(l, predictbert, field="bert")
                mag(l, predicthownet, field="hownet")
                mag(l, predictberthownet, field="hownet_bert")
                # mag(l, predict, field="conceptnet")
            # print(predict[:50])
            # print(predictbert[:50])
            print(name)
            gethuman("200/example.AtLocation.zh.txt", gold)
            score = accuracy_score(gold[:50], predict[:50])
            print("conceptnet", score)
            score = accuracy_score(gold[:50], predictbert[:50])
            print("bert",score)
            score = accuracy_score(gold[:50], predicthownet[:50])
            print("hownet",score)
            score = accuracy_score(gold[:50], predictberthownet[:50])
            print("bert+hownet",score)
            