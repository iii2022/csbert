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

with open("relation.json","r") as a:
    relation=json.load(a);
for k in relation:
    relation[k]=0;
with open("nid_2_triple.json","r") as a:
    nid_2_triple_context=json.load(a);

num=0;


for nid in nid_2_triple_context:
    v=nid_2_triple_context[nid];
    relation[v[3]]=relation[v[3]]+1;
    num=num+1;
print(num)

a=sorted(relation.items(), key=lambda item:item[1])
print(a)