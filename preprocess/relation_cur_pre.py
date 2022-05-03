#sys.setdefaultencoding="utf-8"
import pickle;
import math
import numpy as np
from random import choice
import json
import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math
import numpy as np
import json
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

relation_cur="RelatedTo"
with open("enword2id.json","r") as a:
    enword2id=json.load(a);

with open("word2ids.json","r") as a:
    word2ids=json.load(a);
with open("id2sememeset.json","r") as a:
    id2sememeset=json.load(a);

with open("id2triplelist.json","r") as a:
    id2triplelist=json.load(a);
with open("nid_2_triple.json","r") as a:
    nid_2_triple_context=json.load(a);

with open("relation.json","r") as a:
    relation_all=json.load(a);

H=[];
T=[];
Triple=set();
nid_2_triple_context_used={}
p2w={};
for nid in nid_2_triple_context:
    v=nid_2_triple_context[nid];
    if(v[3]==relation_cur):
        nid_2_triple_context_used[nid]=v;
        H.append(v[4])
        T.append(v[5])
        Triple.add((v[4],v[5],v[3]));
        p2w[v[4]]=v[1];
        p2w[v[5]]=v[2];


with open("nid_2_triple_relation_cur.json","w") as a:
    json.dump(nid_2_triple_context_used,a);
print("当前关系的正三元组数")
print(len(nid_2_triple_context_used))
id=0;
neg={}
for nid in nid_2_triple_context_used:
    v=nid_2_triple_context_used[nid]
    h1=choice(H)

    if((h1,v[5],v[3]) not in Triple):
        v2=v.copy();
        v2[4]=h1;
        v2[1]=p2w[h1];
        neg[id]=v2;
        id+=1;

    t1 = choice(T)
    if ((v[4], t1, v[3]) not in Triple):
        v3 = v.copy();
        v3[5] = t1;
        v3[2]=p2w[t1];
        neg[id] = v3;
        id += 1;

print("当前关系的负三元组数")
print(len(neg))
with open("neg1.json", "w") as a:
    json.dump(neg, a);