import pickle
import json;
import numpy as np
with open("word2sememe", 'rb') as relation0:
    word2sememe=pickle.load(relation0)

s2="en_conceptnet_2";
f2 =open(s2, "r", errors="ignore");
s3="en_conceptnet_3";
f3 =open(s3, "w", errors="ignore");
r_set=set()
c_set=set()
num_joint=0
num_pos=0

while(1):
    line= f2.readline();
    if (len(line) == 0):
        break;
    items=line.strip().split("\t")
    head=items[1]
    tail=items[2]
    r=items[0]
    r_set.add(r.lower())
    c_set.add(head)
    c_set.add(tail)
    if(head.find("/")>0):
        head=head.split("/")[0]
        num_pos+=1
    if(tail.find("/")>0):
        tail=tail.split("/")[0]
        num_pos+=1
    if(head in word2sememe or tail in word2sememe):
        f3.write(line)



for word in word2sememe:
    if(word.lower() in c_set):
        num_joint+=1


print("关系数",len(r_set))
print("概念数",len(c_set))
print("交集",num_joint)
print("有词性的概念数",num_pos)

with open("c_set", 'wb') as relation0:
    pickle.dump(c_set,relation0)




