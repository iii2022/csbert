import pickle
from commons.util import *
from tqdm import tqdm

def split_triple():
    with open("convert_hownet.pkl","rb") as f:
        hownet = pickle.load(f)

    aligment = []
    with open("conceptnet.hownet/HasSubevent_match_result.txt","r") as f:
        for d in tqdm(parse_alignment_noxiaoqi(f,context_f=True)):
            if d["start_id"] is not None:
                start_ids = d["start_id"].split("@@@@")
                for si,sid in enumerate(start_ids):
                    if sid == "":
                        continue
                    start_content = hownet[sid]
                    if si == 0:
                        d["start_synset"] = [start_content["triples"]]
                        d["start"] = start_content["label"]
                    else:
                        if "start_synset" not in d:
                            d["start_synset"] = []
                        d["start_synset"].append(start_content["triples"])
                        d["start"] = d["start"] +"@@@@"+ start_content["label"]
            if d["end_id"] is not None:
                end_ids = d["end_id"].split("@@@@")
                for ei,eid in enumerate(end_ids):
                    if eid == "":
                        continue
                    end_content = hownet[eid]
                    if ei == 0:
                        d["end_synset"] = [end_content["triples"]]
                        d["end"] = end_content["label"]
                    else:
                        if "end_synset" not in d:
                            d["end_synset"] = []
                        d["end_synset"].append(end_content["triples"])
                        d["end"] = d["end"] +"@@@@"+ end_content["label"]
            # d["subject"] = 
            aligment.append(d)

    with open("conceptnet.hownet/convert_aligment_HasSubevent_match_result.pkl", 'wb') as fpkl:
        pickle.dump(aligment, fpkl)

if __name__ == "__main__":
	split_triple()
