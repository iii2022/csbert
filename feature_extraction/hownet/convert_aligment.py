import pickle
from commons.util import *
from tqdm import tqdm

def split_triple():
    with open("convert_hownet.pkl","rb") as f:
        hownet = pickle.load(f)

    aligment = []
    with open("conceptnet.hownet/UsedFor_match_result.txt","r") as f:
        for d in tqdm(parse_alignment(f,context_f=True)):
            if d["start_id"] is not None:
                start_content = hownet[d["start_id"]]
                d["start_synset"] = start_content["triples"]
                d["start"] = start_content["label"]
            if d["end_id"] is not None:
                end_content = hownet[d["end_id"]]
                d["end_synset"] = end_content["triples"]
                d["end"] = end_content["label"]
            # d["subject"] = 
            aligment.append(d)

    with open("conceptnet.hownet/convert_aligment_UsedFor_match_result.pkl", 'wb') as fpkl:
        pickle.dump(aligment, fpkl)

if __name__ == "__main__":
	split_triple()
