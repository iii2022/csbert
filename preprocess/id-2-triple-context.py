import pandas as pd
import os
import json
with open("nid_2_context.json","r") as a:
    nid_2_context=json.load(a);
with open("nid_2_triple.json","r") as a:
    nid_2_triple=json.load(a);
print(len(nid_2_context))
print(len(nid_2_triple))

nid_2_triple_context={}
for nid in nid_2_triple:
    if nid in nid_2_context:
        triple_context=[];
        triple_context.append(nid_2_context[nid][0].replace("[","").replace("]",""))
        triple_context.append(nid_2_context[nid][3])
        triple_context.append(nid_2_context[nid][4])
        triple_context.append(nid_2_triple[nid][1])
        triple_context.append(nid_2_triple[nid][0])
        triple_context.append(nid_2_triple[nid][2])
        nid_2_triple_context[nid]=triple_context;
print(len(nid_2_triple_context))
with open("nid_2_triple_context.json","w") as a:
    json.dump(nid_2_triple_context,a);