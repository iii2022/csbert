import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math
import numpy as np
import json
v_dict={}
with open("1Attributevalue.txt","r") as a:
    line=a.readline().strip().lower();
    while(line!=""):
        b=line.split("    &&    ");
        v=b[0]
        if v not in v_dict:
            v_dict[v] = []
        if(len(b)>1):
            t=b[1].split("    ");

            if(t[2] in v_dict):
                v_dict[t[2]].append(t[0])
        line=a.readline().strip().lower()
v_dict2={}
for k in v_dict:
    k2=k.replace("value","").replace("值","");
    v_dict2[k2]=v_dict[k]

with open("value.json","w") as a:
    json.dump(v_dict2,a,ensure_ascii=False);


