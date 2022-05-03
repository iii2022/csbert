from __future__ import division
import pickle
import json;
import numpy as np
with open("/Users/name/PycharmProjects/Sememe Tree/sememe_2000_embedding", 'rb') as relation0:
    sememe_2000_embedding = pickle.load(relation0);

word2ids={}
id2sememeset={}
id2triplelist={}
with open("1HowNet.json","r") as Hownet_json_target:
    hownet_json = json.load(fp=Hownet_json_target)
    for id in hownet_json:
        data=hownet_json[id]["DEF"]
        word=hownet_json[id]["W_E"].lower()
        if word not in word2ids:
            word2ids[word]=[]
        word2ids[word].append(id);
        id2sememeset[id]=[];
        triple=[];
        for item in data:
            k1 = item.find("{") + 1;
            k2 = item.find("}")
            k3 = k2 + 5;
            k5 = item.rfind("{")
            k4 = k5 - 4;
            k5 += 1;
            k6 = item.rfind("}")
            head = item[k1:k2]
            body = item[k3:k4]
            tail = item[k5:k6]
            t=(head,body,tail);
            triple.append(t);

            id2sememeset[id].append(body)
            if head in sememe_2000_embedding:
                loa=head.find("|")
                id2sememeset[id].append(head[0:loa])
            if tail in sememe_2000_embedding:
                loa = tail.find("|")
                id2sememeset[id].append(tail[0:loa])
        id2triplelist[id]=triple;

with open("word2ids.json","w") as a:
    json.dump(word2ids,a);
    print(len(word2ids))
with open("id2sememeset.json","w") as a:
    json.dump(id2sememeset,a);
    print(len(id2sememeset))


with open("id2triplelist.json","w") as a:
    json.dump(id2triplelist,a,ensure_ascii=False);
    print(len(id2triplelist))