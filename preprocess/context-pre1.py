import pandas as pd
import os
import json

dirs="/Users/name/PycharmProjects/Hownet-Conceptnet/context";
myList=os.listdir(dirs)
print(myList)
nid_2_context={}
for dir_i in myList:
    dir_j="context/"+dir_i;
    data = pd.read_csv(dir_j)
    print(data.values[1])
    for line in data.values:

        id=line[0];


        v=line[1:6];

        v[1]=v[1].lower()
        v[2] = v[2].lower()
        v[3] = str(v[3]).lower()
        v[4] = str(v[4]).lower()
        context=[];
        for j in range(5):
            context.append(v[j]);
        nid_2_context[id]=context;

print(len(nid_2_context))
with open("nid_2_context.json","w") as a:
    json.dump(nid_2_context,a,ensure_ascii=False);


