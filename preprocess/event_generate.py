from __future__ import division
import pickle
import json;
import numpy as np

with open("/Users/name/PycharmProjects/Sememe Tree/sememe_2000_embedding", 'rb') as relation0:
    sememe_2000_embedding = pickle.load(relation0);

e_dict={}
with open("1Event.txt","r") as a:
    line=a.readline().strip().lower().replace("{","").replace("}","");
    while(line!=""):
        b=line.split("    &&    ");
        v=b[0]
        if v not in e_dict:
            e_dict[v] = []
        line=a.readline().strip().lower().replace("{","").replace("}","")

with open("1Entity.txt","r") as a:
    line=a.readline().strip().lower().replace("{","").replace("}","");
    while(line!=""):
        b=line.split("    &&    ");
        for i in range(1,len(b)):
            t=b[i].split("    ");
            if t[2] in e_dict :
                if t not in e_dict[t[2]]:
                    e_dict[t[2]].append(t)
            if t[0] in e_dict :
                if t not in e_dict[t[0]]:
                    e_dict[t[0]].append(t)
        line=a.readline().strip().lower().replace("{","").replace("}","")

with open("1HowNet.json","r") as Hownet_json_target:
    hownet_json = json.load(fp=Hownet_json_target)
    for id in hownet_json:
        data=hownet_json[id]["DEF"]
        word=hownet_json[id]["W_E"].lower()
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
            if head in e_dict and tail in sememe_2000_embedding:
                if t not in e_dict[head]:
                    e_dict[head].append(t)

            if tail in e_dict and head in sememe_2000_embedding:
                if t not in e_dict[tail]:
                    e_dict[tail].append(t)


print(e_dict)

with open("event.json","w") as a:
    json.dump(e_dict,a,ensure_ascii=False);
