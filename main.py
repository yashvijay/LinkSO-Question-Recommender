import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import json
from os import listdir
from os.path import isfile, join
from collections import defaultdict

from gensim import corpora, models, similarities
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

datadir="data/linkSO"
qtype='python'
lamda=10
fields=['qHeader','qDescription','topVotedAnswer']
coeffs=[0.4,0.5,0.1]

class MyCorpus(object):
	def __init__(self,cpath,dictionary,documents=[],rep='bow'):
		self.rep='bow'
		self.cpath=cpath
		self.dict = dictionary
		if len(documents)>0:
			with open(cpath,'w') as f:
				for d in documents:
					f.write(d+"\n")

	def __iter__(self):
		for line in open(self.cpath):
			if self.rep=='bow':
				yield self.dictionary.doc2bow(d.split())

def prepare_corpus(datadir,qtype='python',recompute=False):
	pdf = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_qid2all.txt"), sep ='\t', \
	                    names = ['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
	pdf['type']= qtype
	if isfile(join(datadir,qtype+"_qa.dict")) and (recompute==False):
		dictionary=corpora.Dictionary.load(join(datadir,qtype+"_qa.dict"))
		with open(join(datadir,qtype+"_pwC.json"),'r') as f:
			pwC=json.load(f)
		#corpus = MyCorpus(join(datadir,qtype+"_corpus.txt"),dictionary)
	else:

		pdf.loc[:,'text']=pdf['qHeader'].astype(str)+pdf['qDescription'].astype(str)+pdf['topVotedAnswer'].astype(str)

		documents = pdf['text']
		frequency = defaultdict(int)
		numtokens=0
		for d in documents:
		    for token in d.split():
		        frequency[token]+=1
		        numtokens+=1

		bowdocs = [d.split() for d in documents]
		dictionary = corpora.Dictionary(bowdocs)
		dictionary.save(join(datadir,qtype+"_qa.dict")) 

		#corpus = MyCorpus(join(datadir,qtype+"_corpus.txt"),dictionary,documents=documents)
		pwC={}
		for token in frequency:
			pwC[dictionary.token2id[token]]=frequency[token]/numtokens
		with open(join(datadir,qtype+"_pwC.json"),'w') as f:
			json.dump(pwC,f)

	return dictionary, pwC, pdf

def list2dict(l):
	dt={}
	for a,b in l:
		dt[a]=b
	return dt

def MRR(l):
	for i in range(len(l)):
		if l[i]==1:
			return 1/(i+1)
	return 0

def nDCG(l,n):
	sl = sorted(l,reverse=True)
	a,b=l[0],sl[0]
	for i in range(1,n):
		a+=l[i]/np.log2(i+1)
		b+=sl[i]/np.log2(i+1)
	return a/b

def main():
	dictionary, pwC, pdf = prepare_corpus("data/linkSO",recompute=False)
	valids = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_valid_qid.txt"), sep = '\t',\
	                      names = ['qId'])
	pyscore = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_cosidf.txt"), sep ='\t', \
	                    names = ['qID_1', 'qID_2', 'score', 'label'], skiprows=1)
	valscore = pyscore.merge(valids,left_on='qID_1',right_on='qId',how='inner')
	print (valscore.columns)

	mrr, ndcg5, ndcg10 = 0,0,0
	for i in range(0,len(valscore),30):
		q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
		counts_q = list2dict(dictionary.doc2bow(q['qDescription'].values[0].split()) )
		retrieval_list = []
		for j in range(30):
			qp = pdf.loc[pdf['qID']==valscore.loc[i+j]['qID_2']]
			dataqp,lenqp=[],0
			for f in range(len(fields)):
				doc=str(qp[fields[f]].values[0]).split()
				dataqp.append((coeffs[f],len(doc),list2dict(dictionary.doc2bow(doc))))
				lenqp+=dataqp[-1][1]
			score=0
			for w in counts_q:
				pwQA = sum([(x[0]*x[2].get(w,0))/x[1] for x in dataqp])
				psmooth = (lamda/(lenqp+lamda))*pwC[str(w)]+(lenqp/(lenqp+lamda))*pwQA
				score+=counts_q[w]*np.log(psmooth)
			retrieval_list.append((score,valscore.loc[i+j]['label']))
		retrieval_list = sorted(retrieval_list,key= lambda x: x[0],reverse=True)
		relevance_list = [x[1] for x in retrieval_list]
		mrr+=MRR(relevance_list) 
		ndcg5+=nDCG(relevance_list,5)
		ndcg10+=nDCG(relevance_list,10)
		print ("{}/{}".format(i/30,len(valscore)/30))
	mrr/=(len(valscore)/30)
	ndcg5/=(len(valscore)/30)
	ndcg10/=(len(valscore)/30)
	print ("scores-> mrr={}, ndcg5={}, ndcg10={}".format(mrr,ndcg5,ndcg10))


if __name__ == '__main__':
	main()