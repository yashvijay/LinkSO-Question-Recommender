import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import json
from os import listdir
from gensim.models import TfidfModel
from os.path import isfile, join
from collections import defaultdict
import argparse,sys
import copy
from random import randint
import random
import csv
import translation_lm
from gensim.models import Word2Vec

from gensim import corpora, models, similarities
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepare_corpus(datadir,qtype='python',recompute=False):
	pdf = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_qid2all.txt"), sep ='\t', \
	                    names = ['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
	pdf['type']= qtype
	pdf.loc[:,'text']=pdf['qHeader'].astype(str)+pdf['qDescription'].astype(str)+pdf['topVotedAnswer'].astype(str)
	pdf.loc[:,'question']=pdf['qHeader'].astype(str)+pdf['qDescription'].astype(str)
	documents = pdf['text']
	bowdocs = [d.split() for d in documents]
	l=[len(d) for d in bowdocs]
	avgdl=sum(l)/float(len(l))
	
	if isfile(join(datadir,qtype+"_qa.dict")) and (recompute==False):
		dictionary=corpora.Dictionary.load(join(datadir,qtype+"_qa.dict"))
		tfidf=models.TfidfModel.load(join(datadir,qtype+"_qa.tfidf"))
		with open(join(datadir,qtype+"_pwC.json"),'r') as f:
			pwC=json.load(f)
	else:

		frequency = defaultdict(int)
		numtokens=0
		for d in documents:
		    for token in d.split():
		        frequency[token]+=1
		        numtokens+=1

		dictionary = corpora.Dictionary(bowdocs)
		dictionary.save(join(datadir,qtype+"_qa.dict")) 

		corpus = [dictionary.doc2bow(d) for d in bowdocs]
		tfidf = TfidfModel(corpus)
		tfidf.save(join(datadir,qtype+"_qa.tfidf"))

		pwC={}
		for token in frequency:
			pwC[dictionary.token2id[token]]=frequency[token]/numtokens
		with open(join(datadir,qtype+"_pwC.json"),'w') as f:
			json.dump(pwC,f)

	return dictionary, pwC, pdf, tfidf, avgdl

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

def LDA(dictionary,pdf,path,numtopics=10,recompute=False):
	if isfile(path) and (recompute==False):
		lda = models.LdaModel.load(path)
	else:

		documents = pdf['text']

		bowdocs = [dictionary.doc2bow(d.split()) for d in documents]
		lda = models.LdaModel(bowdocs, num_topics=numtopics)
		lda.save(path)

	return lda

def vecsim(t1,t2):
	d1,d2=list2dict(t1),list2dict(t2)
	simscore=0
	n1,n2=0,0
	for t in d1:
		simscore+=d1[t]*d2.get(t,0)
		n1+=d1[t]**2
	for t in d2:
		n2+=d2[t]**2
	simscore/=np.sqrt(n1*n2)
	return simscore

def cosinesim(a,b):
	return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def compute_scores(retrieval_list):
	mrr, ndcg5, ndcg10 = 0,0,0
	for i in range(0,len(retrieval_list),30):
		ls = sorted(retrieval_list[i:i+30],key= lambda x: x[0],reverse=True)
		ls = [x[1] for x in ls]
		mrr+=MRR(ls) 
		ndcg5+=nDCG(ls,5)
		ndcg10+=nDCG(ls,10)
	mrr/=(len(retrieval_list)//30)
	ndcg5/=(len(retrieval_list)//30)
	ndcg10/=(len(retrieval_list)//30)
	return mrr,ndcg5,ndcg10

class ql_scorer:
	def __init__(self,name,dictionary,pdf,pwC,args):
		self.name=name
		self.pdf=pdf
		self.pwC=pwC
		self.lamda=args['lamda']
		self.coeffs=args['coeffs']
		self.fields=args['fields']
		self.dictionary=dictionary
		if self.name=='translm':
			self.T=translation_lm.load_model_test('data/linkSO/modelQHQD_Model1_python.pk')
		if self.name=='word2vec':
			self.w2vmodel=Word2Vec.load("data/linkSO/{}_word2vec.model".format(args['qtype'])).wv


	def train(self):
		return

	def rank(self,valscore):
		numdocs=len(valscore)//30
		pdf,pwC,lamda,coeffs,fields=self.pdf,self.pwC,self.lamda,self.coeffs,self.fields
		dictionary=self.dictionary
		retrieval_list=[]
		passlist,faillist=[],[]
		for i in range(0,numdocs*30,30):
			q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
			counts_q = list2dict(dictionary.doc2bow(q['question'].values[0].split()) )
			scorelist=[]
			for j in range(30):
				qp = pdf.loc[pdf['qID']==valscore.loc[i+j]['qID_2']]
				dataqp,lenqp=[],0
				for f in range(len(fields)):
					doc=str(qp[fields[f]].values[0]).split()
					dataqp.append((coeffs[f],len(doc),list2dict(dictionary.doc2bow(doc))))
					lenqp+=dataqp[-1][1]
				score=0
				for w in counts_q:
					if self.name=='vanilla':
						pwQA = sum([(x[0]*x[2].get(w,0))/x[1] for x in dataqp])
					if self.name=='translm':
						pwQA = 0
						for itr in range(3):
							tmp=0
							for t in dataqp[itr][2]:
								tmp+=dataqp[itr][2][t]*self.T[dictionary[w]][dictionary[t]]
							pwQA+=dataqp[itr][0]*tmp/dataqp[itr][1]
					if self.name=='word2vec':
						pwQA = 0
						for itr in range(3):
							tmp=0
							for t in dataqp[itr][2]:
								if dictionary[w] in self.w2vmodel.vocab and dictionary[t] in self.w2vmodel.vocab:
									tmp+=dataqp[itr][2][t]*cosinesim(self.w2vmodel[dictionary[w]],self.w2vmodel[dictionary[t]])
							tmp=max(tmp,0)
							pwQA+=dataqp[itr][0]*tmp/dataqp[itr][1]
					psmooth = (lamda/(lenqp+lamda))*pwC[str(w)]+(lenqp/(lenqp+lamda))*pwQA
					score+=counts_q[w]*np.log(psmooth)
				retrieval_list.append((score,valscore.loc[i+j]['label']))
				scorelist.append((score,valscore.loc[i+j]['label'],j))
			scorelist=sorted(scorelist,key= lambda x: x[0],reverse=True)
			for ind,ele in list(enumerate(scorelist)):
				if ele[1]==1:
					passlist.append((ind,q['question'].values[0],pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0]))
					break
			for ind,ele in reversed(list(enumerate(scorelist))):
				if ele[1]==1:
					faillist.append((ind,q['question'].values[0],pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0]))
					break
				
			if (i//30)%(numdocs//10)==0:
				print (i)
		passlist=sorted(passlist,key=lambda x: x[0])[:10]
		faillist=sorted(faillist,key=lambda x: x[0],reverse=True)[:10]
		return retrieval_list,passlist,faillist

class dg_scorer:
	def __init__(self,name,tfidf,dictionary,pdf,args,avgdl):
		self.name=name
		self.pdf=pdf
		self.k1=args['k1']
		self.k3=args['k3']
		self.b=args['b']
		self.dictionary=dictionary
		self.tfidf=tfidf
		self.avgdl=avgdl


	def train(self):
		return

	def rank(self,valscore):
		numdocs=len(valscore)//30
		pdf,k1,k3,b,tfidf=self.pdf,self.k1,self.k3,self.b,self.tfidf
		dictionary,avgdl=self.dictionary,self.avgdl
		retrieval_list=[]
		passlist,faillist=[],[]
		for i in range(0,numdocs*30,30):
			q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
			counts_q = list2dict(dictionary.doc2bow(q['question'].values[0].split()))
			#tfidf_q = list2dict(tfidf[q] )
			scorelist=[]
			for j in range(30):
				qp = pdf.loc[pdf['qID']==valscore.loc[i+j]['qID_2']]
				counts_qp = list2dict(dictionary.doc2bow(qp['text'].values[0].split()))
				len_qp = len(qp['text'].values[0].split())
				score=0
				for w in counts_q:
					if self.name=='rsj':
						if w in counts_qp:
							#score+=tfidf_q[w]/counts_q[w]
							score+=tfidf.idfs[w]
					if self.name=='tfidf':
						if w in counts_qp:
							#score+=tfidf_q[w]
							score+=tfidf.idfs[w]*counts_q[w]
					if self.name=='bm25':
						#score+=tfidf_q[w]*counts_qp.get(w,0)*(k+1)/(counts_qp.get(w,0)+k*(1-b+b*len_qp/avgdl))
						score+=tfidf.idfs[w]*counts_qp.get(w,0)*(k1+1)/(counts_qp.get(w,0)+k1*(1-b+b*len_qp/avgdl))
					if self.name=='bm255':
						score+=tfidf.idfs[w]*(counts_qp.get(w,0)*(k1+1)/(counts_qp.get(w,0)+k1*(1-b+b*len_qp/avgdl)))*counts_q[w]*(k3+1)/(counts_q[w]+k3)
				retrieval_list.append((score,valscore.loc[i+j]['label']))
				scorelist.append((score,valscore.loc[i+j]['label'],j))
			scorelist=sorted(scorelist,key= lambda x: x[0],reverse=True)
			for ind,ele in list(enumerate(scorelist)):
				if ele[1]==1:
					# print(ind)
					# print (q['question'].values[0])
					# print (pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0])
					passlist.append((ind,q['question'].values[0],pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0]))
					break
			for ind,ele in reversed(list(enumerate(scorelist))):
				if ele[1]==1:
					faillist.append((ind,q['question'].values[0],pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0]))
					break
			if (i//30)%(numdocs//10)==0:
				print (i)
		passlist=sorted(passlist,key=lambda x: x[0])[:10]
		faillist=sorted(faillist,key=lambda x: x[0],reverse=True)[:10]
		return retrieval_list,passlist,faillist

class topic_scorer:
	def __init__(self,dictionary,pdf,args):
		self.pdf=pdf
		self.dictionary=dictionary
		self.args=args

	def train(self,args,numtopics=100):
		path=join("ldaModels","{}_{}".format(numtopics,self.args['qtype']))
		self.lda=LDA(self.dictionary,self.pdf,path,numtopics=numtopics)

	def rank(self,valscore):
		mrr, ndcg5, ndcg10 = 0,0,0
		numdocs=len(valscore)//30
		pdf,lda,dictionary=self.pdf,self.lda,self.dictionary
		retrieval_list = []
		passlist,faillist=[],[]
		for i in range(0,numdocs*30,30):
			q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
			qtopics = lda[dictionary.doc2bow(q['question'].values[0].split())]
			scorelist=[]
			for j in range(30):
				qp = pdf.loc[pdf['qID']==valscore.loc[i+j]['qID_2']]
				qptopics = lda[dictionary.doc2bow(qp['text'].values[0].split())]
				score=vecsim(qtopics,qptopics)
				retrieval_list.append((score,valscore.loc[i+j]['label']))
				scorelist.append((score,valscore.loc[i+j]['label'],j))
			scorelist=sorted(scorelist,key= lambda x: x[0],reverse=True)
			for ind,ele in list(enumerate(scorelist)):
				if ele[1]==1:
					passlist.append((ind,q['question'].values[0],pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0]))
					break
			for ind,ele in reversed(list(enumerate(scorelist))):
				if ele[1]==1:
					faillist.append((ind,q['question'].values[0],pdf.loc[pdf['qID']==valscore.loc[i+ele[2]]['qID_2']]['text'].values[0]))
					break
			if (i//30)%(numdocs//10)==0:
				print (i)
		passlist=sorted(passlist,key=lambda x: x[0])[:10]
		faillist=sorted(faillist,key=lambda x: x[0],reverse=True)[:10]
		return retrieval_list,passlist,faillist


def main(args):
		
	with open("quantitative_analysis.csv",'w') as f:
		f.write("")

	with open("quantitative_analysis.csv",'a') as f:
		writer = csv.writer(f)
		writer.writerow(['qtype','method','params','mrr','ndcg5','ndcg10'])

		datadir=args['datadir']

		for qtype in ['java','javascript','python']:
			args['qtype']=qtype

			dictionary, pwC, pdf, tfidf, avgdl = prepare_corpus(datadir,qtype=qtype,recompute=False)

			testids = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_valid_qid.txt"), sep = '\t',\
			                      names = ['qId'])
			pyscore = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_cosidf.txt"), sep ='\t', \
			                    names = ['qID_1', 'qID_2', 'score', 'label'], skiprows=1)
			testscore = pyscore.merge(testids,left_on='qID_1',right_on='qId',how='inner')

			#generating results for query generation
			#for method in ['word2vec','vanilla','translm']:
			# for method in ['vanilla','translm']:
			# 	if args['qtype']!='python' and method=='translm':
			# 		continue

			# 	s1=ql_scorer(method,dictionary,pdf,pwC,args)
			# 	s1.train()
			# 	rl1,var1,var2=s1.rank(testscore)
			# 	if qtype=='python':
			# 		with open("quals_{}.txt".format(method),'w') as f:
			# 			f.write("positive samples:\n")
			# 			for r,q1,q2 in var1:
			# 				f.write("{}    -------    {}    ------   {}\n".format(r,q1,q2))
			# 			f.write("negative samples:\n")
			# 			for r,q1,q2 in var2:
			# 				f.write("{}    -------    {}    ------   {}\n".format(r,q1,q2))
			# 	mrr, ndcg5, ndcg10 = compute_scores(rl1)
			# 	c1,c2,c3=args['coeffs']
			# 	print ([qtype,method,'lamda={},c1={},c2={},c3={}'.format(args['lamda'],c1,c2,c3),mrr,ndcg5,ndcg10])
			# 	writer.writerow([qtype,method,'lamda={},c1={},c2={},c3={}'.format(args['lamda'],c1,c2,c3),mrr,ndcg5,ndcg10])

			#generating results for topic matching
			# mrr,ndcg5,ndcg10=3,4,5
			# for numtopics in [10,50,100,150,200]:
			# 	s2=topic_scorer(dictionary,pdf,args)
			# 	s2.train(args,numtopics=numtopics)
			# 	rl2,var1,var2=s2.rank(testscore)
			# 	if numtopics==10 and qtype=='python':
			# 		with open("quals_topics.txt",'w') as f:
			# 			f.write("positive samples:\n")
			# 			for r,q1,q2 in var1:
			# 				f.write("{}    -------    {}    ------   {}\n".format(r,q1,q2))
			# 			f.write("negative samples:\n")
			# 			for r,q1,q2 in var2:
			# 				f.write("{}    -------    {}    ------   {}\n".format(r,q1,q2))
			# 	mrr, ndcg5, ndcg10 = compute_scores(rl2)
			# 	print ([qtype,'topics','numtopics={}'.format(numtopics),mrr,ndcg5,ndcg10])
			# 	writer.writerow([qtype,'topics','numtopics={}'.format(numtopics),mrr,ndcg5,ndcg10])

			#generating results for document generation
			for method in ['rsj','tfidf','bm25','bm255']:
				s1=dg_scorer(method,tfidf,dictionary,pdf,args,avgdl)
				s1.train()
				rl1,var1,var2=s1.rank(testscore)
				if qtype=='python':
					with open("quals_{}.txt".format(method),'w') as f:
						f.write("positive samples:\n")
						for r,q1,q2 in var1:
							f.write("{}    -------    {}    ------   {}\n".format(r,q1,q2))
						f.write("negative samples:\n")
						for r,q1,q2 in var2:
							f.write("{}    -------    {}    ------   {}\n".format(r,q1,q2))
				mrr, ndcg5, ndcg10 = compute_scores(rl1)
				print ([qtype,method,'',mrr,ndcg5,ndcg10])
				writer.writerow([qtype,method,'',mrr,ndcg5,ndcg10])



			#tuning lamda and coeffs in qg
			# for lamda in [0.000001,0.00001,0.0001,0.001,0.01]:
			# 	args['lamda']=lamda
			# 	s1=ql_scorer('vanilla',dictionary,pdf,pwC,args)
			# 	s1.train()
			# 	rl1,var1,var2=s1.rank(testscore)
			# 	mrr, ndcg5, ndcg10 = compute_scores(rl1)
			# 	c1,c2,c3=args['coeffs']
			# 	print ([qtype,'vanilla','lamda={},c1={},c2={},c3={}'.format(lamda,c1,c2,c3),mrr,ndcg5,ndcg10])
			# 	writer.writerow([qtype,'vanilla','lamda={},c1={},c2={},c3={}'.format(lamda,c1,c2,c3),mrr,ndcg5,ndcg10])

			# args['lamda']=0.001
			# coeffs=[5,5,5]
			# for _ in range(5):
			# 	args['coeffs']=copy.deepcopy(coeffs)
			# 	args['coeffs'][0]=randint(1,20)
			# 	args['coeffs'][1]=randint(50,80)
			# 	args['coeffs'][2]=randint(20,40)
			# 	s=sum(args['coeffs'])
			# 	l=[str(x) for x in args['coeffs']]
			# 	args['coeffs']=[x/s for x in args['coeffs']]
			# 	s1=ql_scorer('vanilla',dictionary,pdf,pwC,args)
			# 	s1.train()
			# 	rl1,var1,var2=s1.rank(testscore)
			# 	mrr, ndcg5, ndcg10 = compute_scores(rl1)
			# 	c1,c2,c3=args['coeffs']
			# 	print ([qtype,'vanilla','lamda={},c1={},c2={},c3={}'.format(1,c1,c2,c3),mrr,ndcg5,ndcg10])
			# 	writer.writerow([qtype,'vanilla','lamda={},c1={},c2={},c3={}'.format(1,c1,c2,c3),mrr,ndcg5,ndcg10])
			# args['coeffs']=[0.1,0.6,0.3]




			# for _ in range(50):
			# 	args['k']=random.uniform(0.1,5)
			# 	args['b']=random.uniform(0,1)
			# 	s1=dg_scorer('bm25',tfidf,dictionary,pdf,args,avgdl)
			# 	s1.train()
			# 	rl1=s1.rank(testscore)
			# 	mrr, ndcg5, ndcg10 = compute_scores(rl1)
			# 	writer.writerow([qtype,'bm25','k={},b={}'.format(args['k'],args['b']),mrr,ndcg5,ndcg10])

if __name__ == '__main__':
	args={}
	args['datadir']="data/linkSO"
	args['qtype']='python'
	args['lamda']=0.001
	args['beta']=100
	args['fields']=['qHeader','qDescription','topVotedAnswer']
	args['coeffs']=[0.1,0.6,0.3]
	args['k1']=1.2
	args['k3']=1.2
	args['b']=0.75
	main(args)