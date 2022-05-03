import pandas as pd
from tqdm import tqdm
import argparse
from commons.util import *

import spacy

class NLP(object):
	def __init__(self,name="en_core_web_sm"):
		self.nlp = spacy.load(name)

	def parse(self,sentence):
		doc = self.nlp(sentence)
		for token in doc:
			print(token.i,token.text,token.head.text,[child for child in token.head.children])

	def close(self):
		pass


def process_no_context(s):
    arr = s.split("/")
    # print(arr)
    return re.sub(r"_"," ", arr[3])

def root(model, s):
    s = process_no_context(s)
    # find the root of the tree
    doc = model.nlp(s)
    # for token in doc:
        # print(token.i, token.text, token.head)
    sent = [(token.i, token.text) for token in doc]
    root = doc[0]
    tmp = -100
    while tmp != root.i:
        tmp = root.i
        root = root.head
    return (root.i, root.text, sent)


if __name__ == "__main__":
    # context
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--start", default=None, type=int, required=True, help=" ")
    parser.add_argument("--end", default=None, type=int, required=True, help=" ")
    parser.add_argument("--file_name", default=None, type=str, required=True, help=" ")
    args = parser.parse_args()

    # Need to pay attention to the installation and name of the model
    model = NLP("en_core_web_lg")
    # model = NLP("zh_core_web_lg")
    # s=u"Autonomous cars shift insurance, liability toward manufacturers."
    # res = root(model, s)
    # print(res)
    files = list_files("../hownet/conceptnet.triples/triples/","*.csv")
    print(files)
    res = []
    for file in files:
        df = pd.read_csv(file)
        df.fillna("", inplace = True)
        # print(df.shape) #(2176097, 4)
        res.append(df)
    df = pd.concat(res,axis=0)

    total = []
    for j, (i,row) in tqdm(enumerate(df.iterrows())):
        if i >= args.start and i < args.end:
            # if i< 310390:
            #     continue
            row_dict = row.to_dict()
            # print(row_dict)
            if row["subject"].strip() != "":
                row_dict["subject_root"] = root(model, row["subject"])[1]
            else:
                print("error", row_dict)
                continue
            if row["object"].strip() != "":
                row_dict["object_root"] = root(model, row["object"])[1]
            else:
                print("error", row_dict)
                continue
            # print(row_dict)
            total.append(row_dict)
        if i >= args.end:
            break

    dft = pd.DataFrame(total)
    # dft.rename(columns={'s':'statement_id'}, inplace = True)
    # dft.columns = ['statement_id', 'subject', 'predicate', 'object']
    # dft = dft.sort_values('statement_id')
    dft.to_csv(args.file_name, index= False)
