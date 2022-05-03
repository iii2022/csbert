import pandas as pd
import os
import json

dirs="/Users/name/PycharmProjects/Hownet-Conceptnet/triples.csv";
nid_2_triple={}
data = pd.read_csv(dirs)
print(data.values[1])
for line in data.values:
    id=line[0];
    v=line[1:6];
    v[0] = str(v[0]).lower()
    v[2] = str(v[2]).lower()
    v[3] = str(v[3]).lower()
    v[4] = str(v[4]).lower()
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
    context.append(b);
    context.append(v[3])
    context.append(v[4])
    context.append(b);


    context.append(a);
    context.append(c);
    nid_2_triple[id]=context;

print(len(nid_2_triple))
with open("nid_2_triple.json","w") as a:
    json.dump(nid_2_triple,a);

print("done")