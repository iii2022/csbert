from torchtext.vocab import GloVe
from update_conceptnet import *
import torch, pickle
import pandas as pd

def convert2(s):
    s = s.strip()
    labels = s.split("/")
    if len(labels) == 5:
        return labels[3]
    if len(labels) < 5:
        return labels[-1]
    if len(labels) > 5:
        return labels[3]

def convert3(s):
    return s.strip()[3:]

def process_vocab():
    s = read_dump("conceptnet-assertions-5.7.0.csv.gz")
    
    vocab = {}
    # c = 0
    # for l in conceptnet:
    for c in tqdm(range(34074917)):
        l = next(s).decode().strip()
        # print("====")
        # for k in l.split("\t"):
        #     print(k)
        # if c > 10000:
        #     break
        subject = get_subject(l)
        vocab[subject] = convert2(subject)
        if "/en/" not in subject:
            continue
        obj = get_object(l)
        vocab[obj] = convert2(obj)
        if "/en/" not in obj:
            continue
        predicate =  get_predicate(l)
        vocab[predicate] = convert3(predicate)
            
    return vocab

if __name__ == "__main__":
    print("load GLOVE")
    embedding_glove = GloVe(name='840B', dim=300)
    vector = embedding_glove.vectors
    # print(vector[-1])
    print(vector.shape)
    stoi = embedding_glove.stoi
    print(len(stoi))
    print(embedding_glove[","] == vector[0])
    stoi_list = [k for k,v in sorted(stoi.items(), key=lambda item: item[1])]
    print(stoi_list[:10])

    # con_vocab = [k for k in con_vocab.keys() if k not in stoi]
    print("GENERATE")
    all_vocab = stoi_list + ["[unused1]","[unused2]","[unused3]","[unused4]","[unused5]","[unk]", "[pad]"]
    print(len(all_vocab))
    all_vectors = torch.rand(len(all_vocab), 300)
    all_vectors[:len(stoi_list),:] = vector[:len(stoi_list)]

    print("load conceptnet")
    df = pd.read_csv("myreweight.csv", sep="\t")
    df.fillna("", inplace = True)
    graph = {}
    for i, row in tqdm(df.iterrows()):
        weight = row[2]
        if row[0] not in graph:
            if "/en/" not in row[0]:
                continue
            graph[row[0]] = [(convert2(row[1]), weight)]
            graph[row[1]] = [(convert2(row[0]), weight)]
        else:
            graph[row[0]].append((convert2(row[1]), weight))
            graph[row[1]].append((convert2(row[0]), weight))


    # np.save( "all_embed.npy", {"vectors":all_vectors,"vocab":all_vocab})
    with open("all_embed.npy","wb") as f:
        pickle.dump({"vectors":all_vectors,"vocab":all_vocab, "conceptnet":graph}, f, protocol=4)
    


