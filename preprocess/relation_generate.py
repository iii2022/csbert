import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math
import numpy as np
import json
# re={};
# with open("nid_2_triple_context.json","r") as a:
#     nid_2_triple_context=json.load(a);
#
# with open("nid_2_triple.json","r") as a:
#     nid_2_triple=json.load(a);
# for nid in nid_2_triple:
#
#     line=nid_2_triple[nid]
#     r=line[1]
#     re[r]=""
#
# print(re);
# print(len(re))
#
# with open("relation.json","w") as a:
#     json.dump(re,a);

with open("enword2id.json","r") as a:
    enword2id=json.load(a);

with open("relation.json","r") as a:
    relation=json.load(a);
# for k in relation:
#     v=relation[k].split(" ");
#     relation[k]=v;
#
#
# with open("relation.json","w") as a:
#     json.dump(relation,a);
