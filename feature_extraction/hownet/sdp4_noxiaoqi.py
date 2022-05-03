import networkx as nx
import pylab, pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
# font=FontProperties(fname='/usr/share/fonts/wps-office/DejaVuMathTeXGyre.ttf',size=14)
# matplotlib.rcParams['font.family']='SimHei'
# matplotlib.rcParams['font.sans-serif']=['SimHei']
# matplotlib.rcParams['axes.unicode_minus']=False	 # 正常显示负号
import numpy as np
import json, re, argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from sdp2 import *

def pick_syset(sysets):
	sysets = re.sub(r"'","\"", sysets)
	# print("synset", sysets)
	js = json.loads(sysets)
	res = []
	for tree in js:
		for i,x in enumerate(tree):
			if x[0].lower() not in res:
				res.append(x[0].lower())
			if x[2].lower() not in res:
				res.append(x[2].lower())
	return res

def get_sub_edges(path, dct):
	# print(path)
	# print("source:{} -> target:{}".format(s,t))
	G = []
	for i in range(len(path)):
		if i+1 < len(path):
			triple = hash_search(path[i],path[i+1],dct)
			G.append(triple)
	return G
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--start", default=None, type=int, required=True, help=" ")
	parser.add_argument("--end", default=None, type=int, required=True, help=" ")
	parser.add_argument("--file_name", default=None, type=str, required=True, help=" ")

	args = parser.parse_args()
	
	edges, vocab = convert_ontology()
	edges2 = hownet_tree(vocab)
	print(len(edges2))
	edges3 = parse_event()
	print(len(edges3))
	edges4 = parse_value()
	print(len(edges4))
	edges = edges + edges2 + edges3 + edges4
	eds,dct = system_edges(edges)

	eds,dct = system_edges(edges)
	# print(edges[0])
	G=nx.DiGraph()
	for i,e in enumerate(eds):
		# if i>1000 and i < 1800:
		if i >= -1:
			G.add_weighted_edges_from([(e[0],e[2],1)], label=e[1])
	
	print('dijkstra方法寻找最短路径：')
	
	df = pd.read_csv("HasContext_pn.csv")
	df.fillna("", inplace = True)

	res = []
	for j, (i,row) in tqdm(enumerate(df.iterrows())):
		if i >= args.start and i < args.end:
			paths = []
			if row["start_synset"].strip()!="" and row["end_synset"].strip() != "":
				# print(i,row["start_synset"])
				try:
					starts = pick_syset(row["start_synset"]) 
					# print(i,row["end_synset"])
					ends = pick_syset(row["end_synset"])
				except:
					continue

				pair_ = pairs(starts, ends)
				# paths = []
				# find paths
				for s,e in pair_:
					try:
						path=nx.dijkstra_path(G, source=s, target=e)
						if len(path) > 1:
							paths.append(path)
					except:
						# print("no path {}->{}".format(s,e))
						pass
			
			G1 = []
			for j, path in enumerate(paths):
				G1.append(get_sub_edges(path, dct))
				
			# print(
			# """
			# weight:{}
			# start:{}
			# end:{}
			# context:{}
			# start_head:{}
			# end_head:{}
			# start_synset:{}
			# end_synset:{}
			# """.format(row["weight"], row["start"], row["end"], row["context"], row["start_head"], row["end_head"], row["start_synset"], row["end_synset"]))
			tmp = {}
			tmp["statement_id"] = row["statement_id"]
			tmp["conceptnet_triple"]  = "{}####{}####{}".format(row["subject"], row["predicate"], row["object"])
			tmp["hownet_paths"] = G1
			tmp["label"] = row["weight"]
			tmp["context"] = row["context"]
			tmp["start_synset"] = row["start_synset"]
			tmp["end_synset"] = row["end_synset"]
			res.append(tmp)

	with open(args.file_name,"wb") as f:
		pickle.dump(res, f)
