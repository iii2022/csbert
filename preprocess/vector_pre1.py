import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math
import numpy as np
import json

en_embedding_filename ="en_embedding_300.txt";
enword2id={}
matrix=[]
with open(en_embedding_filename, 'r', encoding='utf-8') as en_embedding:

    id=0;
    line=en_embedding.readline()
    while(line!=""):
        line= line.strip().split();
        if (len(line) == 301):
            word = line[0].strip();
            float_arr = []
            for i in range(1, 300 + 1):
                float_arr.append(float(line[i]))
            regular = math.sqrt(sum([x * x for x in float_arr]))
            vec = []
            for i in range(300):
                vec.append(float(float_arr[i]) / regular)
            matrix.append(np.array(vec));
            enword2id[word]=id;
            id+=1;
        if(id%100000==0):
            print(id)
        line = en_embedding.readline()



np.save("en_matrix.npy", matrix);
with open("enword2id.json", "w") as a:
    json.dump(enword2id, a);

# b = np.load('en_matrix.npy')
# print(b)
# print(123)









