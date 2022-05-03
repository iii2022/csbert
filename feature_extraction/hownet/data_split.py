from sklearn.model_selection import train_test_split
import pandas as pd
import pickle, random
from tqdm import tqdm
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

SEED=1
input = "conceptnet.hownet/HasSubevent.pn.model.fix.pkl"
# input = "model_pn.pkl"
relation="HasSubevent_pandn_instance"

with open(input.format(input),"rb") as f:
    df = pickle.load(f)

pos_res = []
neg_res = []
for i, row in tqdm(enumerate(df)):
    if row["start_synset"].strip() != "" and row["end_synset"].strip() != "":
        # print(rowstatement_id)
        if "node" in str(row["statement_id"]):
            pos_res.append(row) 
        else:
            neg_res.append(row)

print(len(pos_res), len(neg_res))
num = min(len(pos_res), len(neg_res))

random.seed(SEED)
random.shuffle(neg_res)
res = pos_res[:num] + neg_res[:num]
train, test = train_test_split(res, test_size=0.2, random_state=SEED)

train, dev = train_test_split(train, test_size=0.2, random_state=SEED)

with open("{}/train.pkl".format(relation),"wb") as f:
    pickle.dump(train, f)
with open("{}/dev.pkl".format(relation),"wb") as f:
    pickle.dump(dev, f)
with open("{}/test.pkl".format(relation),"wb") as f:
    pickle.dump(test, f)

