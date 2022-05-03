import numpy as np
import matplotlib.pyplot as plt
import pickle
# import required libraries
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
 
#   https://www.askpython.com/python/normal-distribution
# creating the dataset
# data = {'C':20, 'C++':15, 'Java':30,
#         'Python':35}
# courses = list(data.keys())
# values = list(data.values())
with open("stc.pkl", "rb") as f:
    score = pickle.load(f)
# courses = []
# values = []
mean, std = 0,0
max1,min1 = 0,0
for k,v in score.items():
    # print(v)
    # if "UsedFor" not in k:
    #     continue
    x = np.array(v)
    v = x.astype(np.float)
    # for js in v:
    #     courses

    # courses = v
    # values = 
    #Calculate mean and Standard deviation.
    # print(v)
    mean = np.mean(v)
    std = np.std(v)
    max1 = np.amax(v)
    min1 = np.amin(v)
    # break
 
    # Creating the distribution
    # data = np.arange(1,10,0.01)
    print(min1,max1,mean,std)
    factor = 0.5/mean
    data = np.arange(min1 * factor,max1 * factor, 0.01)
    
    pdf = norm.pdf(data , loc = mean * factor , scale = std * factor )
    
    #Visualizing the distribution
    
    sb.set_style('whitegrid')
    sb.lineplot(data, pdf , color = 'black')
    plt.xlabel('Score')
    plt.ylabel('Probability Density')
    plt.title(k[3:])
    
    # fig = plt.figure(figsize = (10, 5))
    
    # # creating the bar plot
    # plt.bar(courses, values, color ='maroon',
    #         width = 0.4)
    
    # plt.xlabel("Courses offered")
    # plt.ylabel("No. of students enrolled")
    # plt.title("Students enrolled in different courses")
    # # plt.show()
    plt.savefig("fig/{}-scale-{}.png".format(k[3:], factor))
    plt.clf()import torch, pickle
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
