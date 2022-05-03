import pandas as pd
import json

from glob import glob

# https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python
def list_files(directory, suffix = '*.txt'):
	# directory/     the dir
	# **/       every file and dir under my_path
	# *.txt     every file that ends with '.txt'
	return sorted(glob(directory + '/**/%s' % suffix, recursive=True))

def convert_triples():
	fs = list_files("./conceptnet.triples","*.json")

	fs = [fs[3]]

	total = []
	for f in fs:
	    with open(f,"r") as h:
	        for l in h:
	            ts = json.loads(l)
	            total += ts

	dft = pd.DataFrame(total)
	dft.rename(columns={'s':'statement_id', 'sub':'subject', 'pre':'predicate','obj':'object'}, inplace = True)
	# dft.columns = ['statement_id', 'subject', 'predicate', 'object']
	# dft = dft.sort_values('statement_id')

	dft.to_csv("triples3.csv", index= False)


def convert_vocab():
	fs = list_files("./conceptnet.vocab","*.json")
	print(fs)
	i = 7
	fs = [fs[i]]
	total = []
	for f in fs:
	    with open(f,"r") as h:
	        for l in h:
	            ts = json.loads(l)
	            total += ts

	dft = pd.DataFrame(total)
	dft.rename(columns={'sub':'uri', 'text':'label'}, inplace = True)
	# dft.columns = ['statement_id', 'subject', 'predicate', 'object']
	# dft = dft.sort_values('statement_id')

	dft.to_csv("vocab{}.csv".format(i), index= False)

def convert_context():
	fs = list_files("./conceptnet.context","*.json")
	print(fs)
	i = 0
	fs = [fs[i]]
	total = []
	for f in fs:
	    with open(f,"r") as h:
	        for l in h:
	            ts = json.loads(l)
	            total += ts

	dft = pd.DataFrame(total)
	dft.rename(columns={'s':'statement_id'}, inplace = True)
	# dft.columns = ['statement_id', 'subject', 'predicate', 'object']
	# dft = dft.sort_values('statement_id')

	dft.to_csv("context{}.csv".format(i), index= False)

def convert_weight():
	fs = list_files("./conceptnet.weight","*.json")
	print(fs)
	i = "all"
	# fs = [fs[i]]
	total = []
	for f in fs:
	    with open(f,"r") as h:
	        for l in h:
	            ts = json.loads(l)
	            total += ts
	print(len(total))
	dft = pd.DataFrame(total)
	dft.rename(columns={'s':'statement_id'}, inplace = True)
	# dft.columns = ['statement_id', 'subject', 'predicate', 'object']
	# dft = dft.sort_values('statement_id')

	dft.to_csv("context{}.csv".format(i), index= False)

if __name__ == "__main__":
	convert_weight()
