import pandas as pd
import json, re

from glob import glob

# https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python
def list_files(directory, suffix = '*.txt'):
	# directory/     the dir
	# **/       every file and dir under my_path
	# *.txt     every file that ends with '.txt'
	return sorted(glob(directory + '/**/%s' % suffix, recursive=True))


def parse_alignment(handler, context_f = True):
	tmp = {}
	for i,l in enumerate(handler):

		if i %3 == 0:
			if context_f:
				tmp = {}
				arr = l.strip().split("@@@@")
				# print(arr)
				# stmt_id, context, subject, object_ = arr[0], arr[1], arr[2], arr[3]
				stmt_id, context_s, subject, object_ = arr[0], None, arr[1].strip(), arr[2].strip()
				tmp["statement_id"] = stmt_id
				tmp["context"] = context_s
				tmp["subject"] = subject
				tmp["object"] = object_
			else:
				tmp = {}
				arr = l.strip().split("  ")
				spo = re.findall(r"\[.*\]",l)
				# print(spo)
				# ss = re.sub(r"\\","", spo[0])
				# print(ss)
				# spo_js = json.loads(re.sub(r"'","\"",ss))
				# print(arr)
				stmt_id = arr[0]
				tmp["statement_id"] = stmt_id
				tmp["context"] = None
				tmp["subject"] = None
				tmp["object"] = None

		elif i%3 == 1:
			arr = l.strip().split()
			# print(arr)
			synsets = re.findall(r"\[.*\]",l)
			if synsets:
				node_id, head, synset = arr[0], arr[1], synsets[0]
			else:
				node_id, head, synset = None, None, None
			tmp["start_id"] = node_id
			tmp["start_head"] = head
			tmp["start_synset"] = synset

		elif i%3 ==2:
			arr = l.strip().split()
			# print(arr)
			synsets = re.findall(r"\[.*\]",l)
			if synsets:
				node_id, head, synset = arr[0], arr[1], synsets[0]
			else:
				node_id, head, synset = None, None, None
			tmp["end_id"] = node_id
			tmp["end_head"] = head
			tmp["end_synset"] = synset
			yield tmp


def parse_alignment_noxiaoqi(handler, context_f = True):
	lines = handler.readlines()
	tmp = {}
	i=0
	# l = "1"
	while i < len(lines):
	# for i,l in enumerate(handler):
		l=lines[i]
		if i % 3 == 0:
			if context_f:
				tmp = {}
				arr = l.strip().split("@@@@")
				# print(arr)
				try:
					stmt_id, context_s, subject, object_ = arr[0], None, arr[1].strip(), arr[2].strip()
				except:
					print("ERROR",arr)
					i = i+3
					continue
				tmp["statement_id"] = stmt_id
				tmp["context"] = context_s
				tmp["subject"] = subject
				tmp["object"] = object_
			else:
				tmp = {}
				arr = l.strip().split("  ")
				spo = re.findall(r"\[.*\]",l)
				# print(spo)
				# ss = re.sub(r"\\","", spo[0])
				# print(ss)
				# spo_js = json.loads(re.sub(r"'","\"",ss))
				# print(arr)
				stmt_id = arr[0]
				tmp["statement_id"] = stmt_id
				tmp["context"] = ""
				tmp["subject"] = ""
				tmp["object"] = ""

		elif i%3 == 1:
			hownet = l.strip().split("@@@@")
			for how in hownet:
				arr = how.strip().split()
				# print(arr)
				synsets = re.findall(r"\[.*\]",how)
				if synsets:
					node_id, head, synset = arr[0], arr[1], synsets[0]
				else:
					node_id, head, synset = "", "", ""
				if "start_id" not in tmp:
					tmp["start_id"] = node_id
				else:
					tmp["start_id"] = tmp["start_id"]+"@@@@"+node_id
				
				if "start_head" not in tmp:
					tmp["start_head"] = head
				else:
					tmp["start_head"] = tmp["start_head"]+"@@@@"+head
				
				if "start_synset" not in tmp:
					tmp["start_synset"] = synset
				else:
					tmp["start_synset"] = tmp["start_synset"]+"@@@@"+synset

		elif i%3 ==2:
			hownet = l.strip().split("@@@@")
			for how in hownet:
				arr = how.strip().split()
				# print(arr)
				synsets = re.findall(r"\[.*\]",how)
				if synsets:
					node_id, head, synset = arr[0], arr[1], synsets[0]
				else:
					node_id, head, synset = "", "", ""
				if "end_id" not in tmp:
					tmp["end_id"] = node_id
				else:
					tmp["end_id"] = tmp["end_id"]+"@@@@"+node_id
				
				if "end_head" not in tmp:
					tmp["end_head"] = head
				else:
					tmp["end_head"] = tmp["end_head"]+"@@@@"+head
				
				if "end_synset" not in tmp:
					tmp["end_synset"] = synset
				else:
					tmp["end_synset"] = tmp["end_synset"]+"@@@@"+synset
			yield tmp
			tmp = {}
		i += 1

