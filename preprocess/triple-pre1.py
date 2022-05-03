import pandas as pd
import os
import json

dirs="/Users/name/PycharmProjects/Hownet-Conceptnet/concept";
myList=os.listdir(dirs)
print(myList)
nid_2_triple={}
for dir_i in myList:
    dir_j="concept/"+dir_i;
    data = pd.read_csv(dir_j)
    print(data.values[1])
    for line in data.values:
        id=line[0];

        v=line[1:4];
        context=[];
        if (v[0].find("/c/en/")<0):
            continue;
        if (v[2].find("/c/en/")<0):
            continue;
        if (v[1].find("/r/")<0):
            continue;
        a=v[0].replace("/c/en/", "");
        b=v[1].replace("/r/", "")
        c=v[2].replace("/c/en/", "");
        a=a.replace("_"," ")
        c=c.replace("_"," ")
        if (a.find("/") > 0):
            a = a[0:a.find("/")]
        if (b.find("/") > 0):
            b = b[0:b.find("/")]
        if (c.find("/") > 0):
            c = c[0:c.find("/")]
        context.append(a);
        context.append(b);
        context.append(c);
        nid_2_triple[id]=context;

print(len(nid_2_triple))
with open("nid_2_triple.json","w") as a:
    json.dump(nid_2_triple,a);