# Different model classes needed for different models
from concurrent.futures import thread
import random
import shutil
from pathlib import Path
import copy
from transformers import AutoModel,BertModel, FeatureExtractionPipeline, BertTokenizer
from zmq import device
import dill as pkl

import numpy as np
import os 
import pandas as pd

def get_seq_encoding_cat():
	AA_TRIPLET_TO_SINGLE = { "ARG": "R", "HIS": "H", "LYS": "K", "ASP": "D", "GLU": "E", "SER": "S", "THR": "T", "ASN": "N", "GLN": "Q", "CYS": "C", "SEC": "U", "GLY": "G", "PRO": "P", "ALA": "A", "VAL": "V", "ILE": "I", "LEU": "L", "MET": "M", "PHE": "F", "TYR": "Y", "TRP": "W", }
	AA_SINGLE_TO_TRIPLET = {v: k for k, v in AA_TRIPLET_TO_SINGLE.items()}

	# 21 amino acids
	AMINO_ACIDS = "RHKDESTNQCUGPAVILMFYW"
	assert len(AMINO_ACIDS) == 21
	assert all([x == y for x, y in zip(AMINO_ACIDS, AA_TRIPLET_TO_SINGLE.values())])
	AMINO_ACIDS_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

	# Pad with $ character
	PAD = "$"
	MASK = "."
	UNK = "?"
	SEP = "|"
	CLS = "*"
	AMINO_ACIDS_WITH_ALL_ADDITIONAL = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS
	AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX = {
		aa: i for i, aa in enumerate(AMINO_ACIDS_WITH_ALL_ADDITIONAL)
	}

	def get_pretrained_bert_tokenizer(path):
		tok = BertTokenizer.from_pretrained(
			path,
			do_basic_tokenize=False,
			do_lower_case=False,
			tokenize_chinese_chars=False,
			unk_token=UNK,
			sep_token=SEP,
			pad_token=PAD,
			cls_token=CLS,
			mask_token=MASK,
			padding_side="right",
		)
		return tok

	outdir='data/deepcat/deepcat-seq-encoding/'
	model_dir="wukevin/tcr-bert"
	device=0
	model = AutoModel.from_pretrained(model_dir)

	tok = get_pretrained_bert_tokenizer(model_dir)
	pipeline = FeatureExtractionPipeline(model, tok, device=device)
	num=0
	filelist=[]
	for file in ['NormalCDR3_test.txt', 'TumorCDR3.txt', 'TumorCDR3_test.txt', 'NormalCDR3.txt']:
		print(file)
		with open('data/deepcat/'+file) as f:
			for seq in f.readlines():
				seq=seq[:-1]
				if num%10000==0:
					print(num)
				num+=1
				# continue
				if type(seq)==str and not os.path.exists(outdir+seq+'.npy'):
					encod=pipeline([' '.join(seq)])
					encod=np.asarray(encod).reshape(len(seq)+2,-1)
					np.save(outdir+seq,encod)
				else:
					pass
	print(num)

def deepcat_to_npy():
	data=[]

	outdir='data/deepcat/'
	for file in ['NormalCDR3_test.txt', 'TumorCDR3.txt', 'TumorCDR3_test.txt', 'NormalCDR3.txt']:
		print(file)
		encodes=[]
		with open('data/deepcat/'+file) as f:
			for seq in f.readlines():
				seq=seq[:-1]
				t=np.load('data/deepcat/deepcat-seq-encoding/'+seq+'.npy')
				encodes.append(copy.deepcopy(t[0]))
				del t
		encodes=np.concatenate(encodes,axis=0)
		np.save(f'{outdir}/{file[:-4]}',encodes)

get_seq_encoding_cat()
deepcat_to_npy()

