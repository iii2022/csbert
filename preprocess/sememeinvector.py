import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math
with open("word2ids", 'rb') as relation0:
    word2id=pickle.load(relation0)


with open("id2sememeset", 'rb') as relation0:
    id2sememeset=pickle.load(relation0)

sememe2bool={}
for id in id2sememeset:
    for sememe in id2sememeset[id]:
        sememe2bool[sememe]=0;

en_embedding_filename ="en_embedding_300.txt";
en_target_filename = "en_train_data";
en_word_embedding = {};
with open(en_embedding_filename, 'r', encoding='utf-8') as en_embedding:
        with open(en_target_filename, 'wb') as en_target:

            en_wordsBuf=en_embedding.readlines();

            en_wordlen = len(en_wordsBuf);
            en_words = {};

            for i in range(0, en_wordlen):
                line= en_wordsBuf[i].strip().split();
                en_words[line[0].strip()] = i;
            index = 0;
            en_Strings = [];
            # f = open("hownet_simple","w",encoding='utf-8')
            for word_en in sememe2bool:
                if (word_en in en_words):
                    if (len(en_wordsBuf[en_words[word_en]].strip().split()) == 301):

                        en_Strings.append(en_wordsBuf[en_words[word_en]]);
                        sememe2bool[word_en]=1;
            # for word_en in word2id:
            #     if (word_en in en_words):
            #         if (len(en_wordsBuf[en_words[word_en]].strip().split()) == 301):
            # 
            #             en_Strings.append(en_wordsBuf[en_words[word_en]]);







            for line in en_Strings:
                arr = line.strip().split()
                word = arr[0].strip();
                float_arr = []
                for i in range(1, 300 + 1):
                    float_arr.append(float(arr[i]))
                regular = math.sqrt(sum([x * x for x in float_arr]))
                word = arr[0].strip()
                vec = []
                for i in range(1, 300 + 1):
                    vec.append(float(arr[i]) / regular)
                en_word_embedding[word]=vec;
            pickle.dump(en_word_embedding,en_target);


with open("sememe2bool", 'wb') as relation0:
    pickle.dump(sememe2bool,relation0)