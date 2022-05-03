import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import json

with open("enword2id.json","r") as a:
    enword2id=json.load(a);
relation_cur="Synonym";

with open("nid_2_triple_relation_cur.json","r") as a:
    nid_2_triple_context=json.load(a);
with open("word2ids.json","r") as a:
    word2ids=json.load(a);
with open("id2sememeset.json","r") as a:
    id2sememeset=json.load(a);

with open("id2triplelist.json","r") as a:
    id2triplelist=json.load(a);

b= np.load('en_matrix.npy')
word2sense={}
lemmatizer_i=0
with open("relation.json","r") as a:
    relation_all=json.load(a);

def get_context_result(context):
    a=np.zeros(300);
    i=0
    for word in relation_all[relation_cur]:
        if word in enword2id:
            i+=1;
            a+=b[enword2id[word]]
    if context in enword2id:
        i += 1;
        a += b[enword2id[context]]
    a=a/float(i)
    return a

def get_sememe_result(id):
    a=np.zeros(300);
    i=0
    sememe_set=id2sememeset[id]
    for word in sememe_set:
        if word in enword2id:
            i+=1;
            a+=b[enword2id[word]]
    a=a/float(i)
    return a
with open(relation_cur+"_match_result.txt",'w',encoding='utf-8') as hownet0:

    num_i=0;
    for nid in nid_2_triple_context:

        line=nid_2_triple_context[nid]
        h=line[1]
        t=line[2]
        if line[4] in word2ids:
            h=line[4]
        if line[5] in word2ids:
            t=line[5]
        if((h not in word2ids) and (lemmatizer.lemmatize(h) in word2ids)):
            lemmatizer_i+=1;
            h=lemmatizer.lemmatize(h)
        if ((t not in word2ids) and (lemmatizer.lemmatize(t) in word2ids)):
            lemmatizer_i+=1;
            t = lemmatizer.lemmatize(t)

        try:
            hownet0.write(nid+"  "+line[0] + "  "+h+"   "+t+"\n")
        except:
            continue;


        if h in word2ids:
            if len(word2ids[h])==1:
                hownet0.write(str(word2ids[h][0])+" "+h+" "+str(id2triplelist[word2ids[h][0]])+"\n")
            else:
                maxi=0;
                score=-10000;

                vec_a = get_context_result(t)
                for j in range(len(word2ids[h])):

                    vec_b=get_sememe_result(word2ids[h][j])
                    score_new=np.dot(vec_a,vec_b)
                    if score_new>score:
                        score=score_new
                        maxi=j

                hownet0.write(str(word2ids[h][j])+" "+h+" "+str(id2triplelist[word2ids[h][j]])+"\n")
        else:
            hownet0.write("No head"+"\n")


        if t in word2ids:
            if len(word2ids[t])==1:
                hownet0.write(str(word2ids[t][0])+" "+t+" "+str(id2triplelist[word2ids[t][0]])+"\n")
            else:
                maxi=0;
                score=-10000;
                vec_a = get_context_result(h)
                for j in range(len(word2ids[t])):
                    vec_b=get_sememe_result(word2ids[t][j])
                    score_new=np.dot(vec_a,vec_b)
                    if score_new>score:
                        score=score_new
                        maxi=j

                hownet0.write(str(word2ids[t][j])+" "+t+" "+str(id2triplelist[word2ids[t][j]])+"\n")
        else:
            hownet0.write("No tail"+"\n")

        num_i += 1;
        if (num_i % 10000 == 0):
            print(num_i)


print(lemmatizer_i)



word2sense={}
lemmatizer_i=0

def get_context_result(context):
    a=np.zeros(300);
    i=0
    for word in context:
        if word in enword2id:
            i+=1;
            a+=b[enword2id[word]]
    a=a/float(i)
    return a

def get_sememe_result(id):
    a=np.zeros(300);
    i=0
    sememe_set=id2sememeset[id]
    for word in sememe_set:
        if word in enword2id:
            i+=1;
            a+=b[enword2id[word]]
    a=a/float(i)
    return a
with open(relation_cur+"_match_result_noxiaoqi.txt",'w',encoding='utf-8') as hownet0:

    num_i=0;
    for nid in nid_2_triple_context:

        line=nid_2_triple_context[nid]
        h=line[1]
        t=line[2]
        if line[4] in word2ids:
            h=line[4]
        if line[5] in word2ids:
            t=line[5]
        if((h not in word2ids) and (lemmatizer.lemmatize(h) in word2ids)):
            lemmatizer_i+=1;
            h=lemmatizer.lemmatize(h)
        if ((t not in word2ids) and (lemmatizer.lemmatize(t) in word2ids)):
            lemmatizer_i+=1;
            t = lemmatizer.lemmatize(t)
        context=line[0]
        context=context.split('\t');
        try:
            hownet0.write(nid+"  "+line[0] + "  "+h+"   "+t+"\n")
        except:
            continue;


        if h in word2ids:
            if len(word2ids[h])==1:
                hownet0.write(str(word2ids[h][0])+" "+h+" "+str(id2triplelist[word2ids[h][0]])+"\n")
            else:
                maxi=0;
                score=-10000;
                vec_a = get_context_result(context)
                for j in range(len(word2ids[h])):


                    hownet0.write(str(word2ids[h][j])+" "+h+" "+str(id2triplelist[word2ids[h][j]])+"@@@@")
                hownet0.write("\n");
        else:
            hownet0.write("No head"+"\n")


        if t in word2ids:
            if len(word2ids[t])==1:
                hownet0.write(str(word2ids[t][0])+" "+t+" "+str(id2triplelist[word2ids[t][0]])+"\n")
            else:
                maxi=0;
                score=-10000;
                vec_a = get_context_result(context)
                for j in range(len(word2ids[t])):


                    hownet0.write(str(word2ids[t][j])+" "+t+" "+str(id2triplelist[word2ids[t][j]])+"@@@@")
                hownet0.write("\n");
        else:
            hownet0.write("No tail"+"\n")

        num_i += 1;
        if (num_i % 10000 == 0):
            print(num_i)


print(lemmatizer_i)





