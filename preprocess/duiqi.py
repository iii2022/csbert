import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math
with open("word2ids", 'rb') as relation0:
    word2id=pickle.load(relation0)
with open("id2sememeset", 'rb') as relation0:
    id2sememeset=pickle.load(relation0)
en_target_filename = "en_train_data_2";
with open(en_target_filename, 'rb') as en_target:
    en_word_embedding =pickle.load(en_target);
concepnet_filename ="train100k.txt";
wordset={}
k1=0;
k2=0
k3=0
k4=0
triples=[]
with open(concepnet_filename, 'r', encoding='utf-8') as en_embedding:
    for i in range(60000):
        line=en_embedding.readline();
        words=line.split('\t');
        if(words[0] in en_word_embedding):
            k1+=1;
        if (words[1] in en_word_embedding):
            k2+= 1;
        if (words[2] in en_word_embedding):
            k3+= 1;

        if (words[0] in en_word_embedding and words[1] in en_word_embedding and words[2] in en_word_embedding):
            k4 += 1;
            triples.append(words)


        wordset[words[0]]=0
        wordset[words[1]]=0
        wordset[words[2]]=0


print(k1,k2,k3,k4)
# for words in triples:
#     if words[1] in word2id:
#         max=0;
#         for id in word2id[words[1]]:
#             num=0;
#             sum=0;
#             avg=0;
#             for sememe in id2sememeset[id]:
#                 if sememe in
