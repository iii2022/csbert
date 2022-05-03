import pickle, json, re
import pandas as pd

def process_no_context(s):
    s = s.strip()
    # print(s)
    if "/" not in s:
        return s.strip()
    arr = s.split("/")
    # print(arr)
    if len(arr) >= 3:
        return re.sub(r"_"," ", arr[3])
    else:
        return re.sub(r"_"," ", arr[0])

def process_uri(s):
    if "/" in s:
        # arr = s.split("/")
        # print(arr)
        s = re.sub(r"/"," ",s)
        return re.sub(r"_"," ", s.strip())
    else:
        return re.sub(r"_"," ", s)

def get_english(s):
	arr = s.split("|")
	if len(arr) > 1:
		if arr[0][0]=="{":
			return arr[0][1:].lower()
	else:
		if s[0] == "{" and s[-1] == "}":
			s = s[1:-1]
		return s.lower()

output = "HasContext_match_result"

with open("conceptnet.hownet/aligment_full_match_result_HasContext.pkl","rb") as f:
    data = pickle.load(f)

res = []
for l in data:
    # if l["start_synset"] is None:
    #     l["start_synset"] = None
    # if l["end_synset"] is None:
    #     l["end_synset"] = None
    # print(l["weight"])
    # if (l["predicate"] == "/r/UsedFor" and
    #  len(re.findall(r",", l["start_synset"])) >= 3 and
    # len(re.findall(r",", l["end_synset"])) >= 3):
    # if l["predicate"] == "/r/UsedFor":
    #     l["subject"] = process_no_context(l["subject"])
    #     l["object"] = process_no_context(l["object"])
    #     res.append(l)

    # 负样本不需要这个关系的判断过滤
    # if l["predicate"] == "/r/UsedFor":
    # if l["predicate"] == "/r/RelatedTo":
    if True:
        l["subject"] = process_no_context(l["subject"].strip())
        l["object"] = process_no_context(l["object"].strip())
        res.append(l)
    
    # l["subject"] = l["subject"].strip()
    # l["object"] = l["object"].strip()
    # res.append(l)


    # print(l)
    # l["subject"] = get_english(l["start"])
    # l["object"] = get_english(l["end"])
    # 看看用哪个
    # l["subject"] = process_uri(l["subject"])
    # l["object"] = process_uri(l["object"])
    # l["predicate"] = "/r/UsedFor"
    # res.append(l)

dft = pd.DataFrame(res)
dft.to_csv("{}.csv".format(output), index= False)

with open("{}.pkl".format(output),"wb") as f:
    pickle.dump(res ,f)
