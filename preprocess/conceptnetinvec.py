import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
import pickle;
import math


en_embedding_filename ="en_embedding_300.txt";
concepnet_filename ="train100k.txt";
en_target_filename = "en_train_data";
with open(en_target_filename, 'rb') as en_target:
    en_word_embedding =pickle.load(en_target);
wordset={}
with open(concepnet_filename, 'r', encoding='utf-8') as en_embedding:
    for i in range(60000):
        line=en_embedding.readline();
        words=line.split('\t');
        wordset[words[0]]=0
        wordset[words[1]]=0
        wordset[words[2]]=0
with open(en_embedding_filename, 'r', encoding='utf-8') as en_embedding:
    en_wordsBuf=en_embedding.readlines();
    en_wordlen = len(en_wordsBuf);
    en_words = {};

    for i in range(0, en_wordlen):
        line= en_wordsBuf[i].strip().split();
        en_words[line[0].strip()] = i;
    index = 0;
    en_Strings = [];
    # f = open("hownet_simple","w",encoding='utf-8')
    for word_en in wordset:
        if (word_en in en_words):
            if (len(en_wordsBuf[en_words[word_en]].strip().split()) == 301):

                en_Strings.append(en_wordsBuf[en_words[word_en]]);
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



with open("en_train_data_2", 'wb') as en_target:
    pickle.dump(en_word_embedding,en_target);