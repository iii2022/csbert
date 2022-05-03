import pickle
from commons.util import *

def get_english(s):
	arr = s.split("|")
	if len(arr) > 1:
		if arr[0][0]=="{":
			return arr[0][1:].lower()
	else:
		if s[0] == "{" and s[-1] == "}":
			s = s[1:-1]
		return s.lower()

def add_dict(dct, s):
	english = get_english(s)
	if english in dct:
		dct[english].append(s)
	else:
		dct[english] = [s]

def convert_ontology():
	fs = list_files("ontology","*.txt")

	vocab = {}
	for i,name in enumerate(fs):
		if "1event_HowNet.txt" in name:
			continue
		with open(name, "r") as f:
			for j,l in enumerate(f):
				if l.strip() == "":
					continue

				if j == 0:
					continue

				triples = l.split("&&")
				for t in triples[1:]:
					# print(t)
					arr = t.strip().split("    ")
					# print(arr)
					if len(arr) == 3:
						# vocab.append([arr[-3], arr[-2], arr[-1].strip()])
						add_dict(vocab, arr[-3])
						add_dict(vocab, arr[-2])
						add_dict(vocab, arr[-1].strip())
					elif len(arr) > 4: #['{spray|洒下}', '0', 'LocationFin', '{*}']
						# edges.append([arr[-4], arr[-2], arr[-1].strip()])
						add_dict(vocab, arr[-4])
						add_dict(vocab, arr[-2])
						add_dict(vocab, arr[-1].strip())
	return vocab

if __name__ == "__main__":
	words = convert_ontology()
	with open("hownet_vocab.pkl","wb") as f:
		pickle.dump(words, f)
