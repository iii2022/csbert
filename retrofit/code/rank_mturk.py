import torch, pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from spearman import *
from scipy import stats
import pandas as pd
from tqdm import tqdm


def stoi(w, vocab):
    w = w.strip().lower()
    w_id = len(vocab) - 2
    if w in vocab:
        w_id = vocab[w]
    return w_id
        
def sim_score(w1_s, w2_s, embed, vocab):
    w1_vector, w2_vector = [], []
    for w1, w2 in zip(w1_s, w2_s):
        w1_id = stoi(w1, vocab)
        w2_id = stoi(w2, vocab)
        # print(w1_id, w2_id)
        w1_vector.append(embed[w1_id])
        w2_vector.append(embed[w2_id])

    # score = cosine_similarity(w1_vector, w2_vector)
    # diag_score = np.diag(score)

    diag_score = -np.sum(np.square(np.array(w1_vector) - np.array(w2_vector)),axis = -1)

    # diag_score = np.sum(np.array(w1_vector) * np.array(w2_vector), axis=-1)
    
    sort_score = np.sort(diag_score)[::-1]
    idx = np.argsort(diag_score)[::-1]
    # print(sort_score)
    return idx, sort_score

if __name__ == "__main__":
    w1_s, w2_s = [], []
    df = pd.read_csv("word-benchmarks-master/word-similarity/monolingual/en/mturk-771.csv")
    df = df.sort_values(by=['similarity'], ascending=False)
    print(df)
    for i, row in tqdm(df.iterrows()):
        w1_s.append(row[1].strip().lower())
        w2_s.append(row[2].strip().lower())

    with open("data-lm4kg/train.pkl", "rb") as f:
        d = pickle.load(f)
    # embed = d["vectors"].detach().cpu().numpy()
    
    ckpt = torch.load("ckpt-lm4kg/15000.pt")
    embed = ckpt['embedding.weight'].detach().cpu().numpy()
    print(embed.shape)
    vocab ={w:i for i,w in enumerate(d["vocab"])}
    idx, _ = sim_score(w1_s, w2_s, embed, vocab)
    print(idx)
    # score = spearmanRankCorrelationCoefficient(idx, np.arange(len(idx)))
    score = stats.spearmanr(idx, np.arange(len(idx)))
    print(score)
