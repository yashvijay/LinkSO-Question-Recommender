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

from gensim import corpora, models, similarities
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepare_corpus(datadir,qtype='python',recompute=False):
	pdf = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_qid2all.txt"), sep ='\t', \
	                    names = ['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
	pdf['type']= qtype
	pdf.loc[:,'text']=pdf['qHeader'].astype(str)+pdf['qDescription'].astype(str)+pdf['topVotedAnswer'].astype(str)
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

def topicsim(t1,t2):
	d1,d2=list2dict(t1),list2dict(t2)
	simscore=0
	for t in d1:
		simscore+=np.log(d1[t])+np.log(d2.get(t,0.001))
		return simscore

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
	def __init__(self,dictionary,pdf,pwC,args):
		self.pdf=pdf
		self.pwC=pwC
		self.lamda=args['lamda']
		self.coeffs=args['coeffs']
		self.fields=args['fields']
		self.dictionary=dictionary


	def train(self):
		return

	def rank(self,valscore):
		numdocs=len(valscore)//30
		pdf,pwC,lamda,coeffs,fields=self.pdf,self.pwC,self.lamda,self.coeffs,self.fields
		dictionary=self.dictionary
		retrieval_list=[]
		for i in range(0,numdocs*30,30):
			q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
			counts_q = list2dict(dictionary.doc2bow(q['qDescription'].values[0].split()) )
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
			if (i//30)%(numdocs//10)==0:
				print (i)
		return retrieval_list

class dg_scorer:
	def __init__(self,name,tfidf,dictionary,pdf,args,avgdl):
		self.name=name
		self.pdf=pdf
		self.k=args['k']
		self.b=args['b']
		self.dictionary=dictionary
		self.tfidf=tfidf
		self.avgdl=avgdl


	def train(self):
		return

	def rank(self,valscore):
		numdocs=len(valscore)//30
		pdf,k,b,tfidf=self.pdf,self.k,self.b,self.tfidf
		dictionary,avgdl=self.dictionary,self.avgdl
		retrieval_list=[]
		for i in range(0,numdocs*30,30):
			q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
			q = dictionary.doc2bow(q['qDescription'].values[0].split())
			counts_q = list2dict(q)
			tfidf_q = list2dict(tfidf[q] )
			for j in range(30):
				qp = pdf.loc[pdf['qID']==valscore.loc[i+j]['qID_2']]
				counts_qp = list2dict(dictionary.doc2bow(qp['text'].values[0].split()))
				len_qp = len(qp['text'].values[0].split())
				score=0
				for w in counts_q:
					if self.name=='rsj':
						if w in counts_qp:
							score+=tfidf_q[w]/counts_q[w]
					if self.name=='tfidf':
						if w in counts_qp:
							score+=tfidf_q[w]
					if self.name=='bm25':
						score+=tfidf_q[w]*counts_qp.get(w,0)*(k+1)/(counts_qp.get(w,0)+k*(1-b+b*len_qp/avgdl))
				retrieval_list.append((score,valscore.loc[i+j]['label']))
			if (i//30)%(numdocs//10)==0:
				print (i)
		return retrieval_list

class topic_scorer:
	def __init__(self,dictionary,pdf,args):
		self.pdf=pdf
		self.dictionary=dictionary

	def train(self,args,numtopics=100):
		path=join("ldaModels",str(numtopics))
		self.lda=LDA(self.dictionary,self.pdf,path,numtopics=numtopics)

	def rank(self,valscore):
		mrr, ndcg5, ndcg10 = 0,0,0
		numdocs=len(valscore)//30
		pdf,lda,dictionary=self.pdf,self.lda,self.dictionary
		retrieval_list = []
		for i in range(0,numdocs*30,30):
			q = pdf.loc[pdf['qID']==valscore.loc[i]['qID_1']]
			qtopics = lda[dictionary.doc2bow(q['qDescription'].values[0].split())]
			for j in range(30):
				qp = pdf.loc[pdf['qID']==valscore.loc[i+j]['qID_2']]
				qptopics = lda[dictionary.doc2bow(qp['text'].values[0].split())]
				score=topicsim(qtopics,qptopics)
				retrieval_list.append((score,valscore.loc[i+j]['label']))
			if (i//30)%(numdocs//10)==0:
				print (i)
		return retrieval_list


def main(args):
	if sys.argv[1]=='showresults':
		############document generation
		with open("docGeneration_results.txt",'r') as f:
			r=json.load(f)
		print ('rsj: {}'.format(r['rsj']))
		print ('tfidf:{}'.format(r['tfidf']))
		ma,ik,ib=0,-1,-1
		for k in r['bm25']['k']:
			if r['bm25']['k'][k][0]>ma:
				ma,ik=r['bm25']['k'][k][0],k
		ma=0
		for b in r['bm25']['b']:
			if r['bm25']['b'][b][0]>ma:
				ma,ib=r['bm25']['b'][b][0],b
		if r['bm25']['k'][ik]>r['bm25']['b'][ib]:
			print ('bm25: {} for k={} and b={}'.format(r['bm25']['k'][ik],ik,0.75))
		if r['bm25']['k'][ik]<r['bm25']['b'][ib]:
			print ('bm25: {} for k={} and b={}'.format(r['bm25']['b'][ib],1.2,ib))
		lk=[k for k in r['bm25']['k']]
		plt.plot(lk,[r['bm25']['k'][k][0] for k in lk])
		plt.plot(lk,[r['bm25']['k'][k][1] for k in lk])
		plt.plot(lk,[r['bm25']['k'][k][2] for k in lk])
		plt.legend(['MRR','nDCG@5','nDCG@10'])
		plt.title('effect of k in bm25')
		plt.xlabel('k')
		plt.savefig("bm25_k.png")
		plt.show()
		lb=[b for b in r['bm25']['b']]
		plt.plot(lb,[r['bm25']['b'][b][0] for b in lb])
		plt.plot(lb,[r['bm25']['b'][b][1] for b in lb])
		plt.plot(lb,[r['bm25']['b'][b][2] for b in lb])
		plt.legend(['MRR','nDCG@5','nDCG@10'])
		plt.title('effect of b in bm25')
		plt.xlabel('b')
		plt.savefig("bm25_b.png")
		plt.show()
		##############query generation
		with open("queryGeneration_results.txt",'r') as f:
			r=json.load(f)
		print ('query generation: {} for lamda=1 and coeffs=[0.33,0.33,0.33]'.format(r['lamda']['1']))
		lk=[k for k in r['lamda']]
		plt.plot(lk,[r['lamda'][k][0] for k in lk])
		plt.plot(lk,[r['lamda'][k][1] for k in lk])
		plt.plot(lk,[r['lamda'][k][2] for k in lk])
		plt.legend(['MRR','nDCG@5','nDCG@10'])
		plt.title('effect of lamda in query generation')
		plt.xlabel('lamda')
		plt.savefig("qg_lamda.png")
		plt.show()
		c=[5,5,5]

		for i in range(3):
			lb=[]
			for j in [1,3,5,7,9]:
				tmp=copy.deepcopy(c)
				tmp[i]=j
				lb.append(str(tmp[0])+"-"+str(tmp[1])+"-"+str(tmp[2]))
			plt.plot(lb,[r['coeff'][b][0] for b in lb])
			plt.plot(lb,[r['coeff'][b][1] for b in lb])
			plt.plot(lb,[r['coeff'][b][2] for b in lb])
			plt.legend(['MRR','nDCG@5','nDCG@10'])
			plt.title('effect of coefficient of {} in query generation'.format(args['fields'][i]))
			plt.xlabel('value')
			plt.savefig("qg_"+str(i)+".png")
			plt.show()
		##################topics
		r=json.load(open("topic_results.txt",'r'))
		print ('topic based: {} for numtopics=90'.format(r['90']))
		lk=[k for k in r]
		plt.plot(lk,[r[k][0] for k in lk])
		plt.plot(lk,[r[k][1] for k in lk])
		plt.plot(lk,[r[k][2] for k in lk])
		plt.legend(['MRR','nDCG@5','nDCG@10'])
		plt.title('effect of number of topics')
		plt.xlabel('number of topics')
		plt.savefig("topics.png")
		plt.show()
		return
		

	datadir=args['datadir']
	qtype=args['qtype']

	dictionary, pwC, pdf, tfidf, avgdl = prepare_corpus(datadir,recompute=False)

	testids = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_valid_qid.txt"), sep = '\t',\
	                      names = ['qId'])
	pyscore = pd.read_csv(join(datadir,"linkso/topublish/"+qtype+"/"+qtype+"_cosidf.txt"), sep ='\t', \
	                    names = ['qID_1', 'qID_2', 'score', 'label'], skiprows=1)
	testscore = pyscore.merge(testids,left_on='qID_1',right_on='qId',how='inner')


	# s1=dg_scorer('bm25',tfidf,dictionary,pdf,args,avgdl)
	# s1.train()
	# rl1=s1.rank(testscore)
	# mrr, ndcg5, ndcg10 = compute_scores(rl1)
	# print ("mrr,ndcg5,ndcg10 for ql_scorer: {} {} {}".format(mrr,ndcg5,ndcg10))

	# s1=ql_scorer(dictionary,pdf,pwC,args)
	# s1.train()
	# rl1=s1.rank(testscore)
	# mrr, ndcg5, ndcg10 = compute_scores(rl1)
	# print ("mrr,ndcg5,ndcg10 for ql_scorer: {} {} {}".format(mrr,ndcg5,ndcg10))

	# s2=topic_scorer(dictionary,pdf,args)
	# s2.train(args,numtopics=10)
	# rl2=s2.rank(testscore)
	# mrr, ndcg5, ndcg10 = compute_scores(rl2)
	# print ("mrr,ndcg5,ndcg10 for topic_scorer: {} {} {}".format(mrr,ndcg5,ndcg10))

	# rl3=[(rl1[i][0]+args['beta']*rl2[i][0],rl1[i][1]) for i in range(rl1)]
	# mrr, ndcg5, ndcg10 = compute_scores(rl3)
	# print ("mrr,ndcg5,ndcg10 for combination: {} {} {}".format(mrr,ndcg5,ndcg10))	

	if sys.argv[1]=='topics':
		results={}
		for numtopics in range(10,100,10):
			s2=topic_scorer(dictionary,pdf,args)
			s2.train(args,numtopics=numtopics)
			rl2=s2.rank(testscore)
			mrr, ndcg5, ndcg10 = compute_scores(rl2)
			results[numtopics]=(mrr,ndcg5,ndcg10)
		with open("topic_results.txt",'w') as f:
			json.dump(results,f)

	if sys.argv[1]=='qg':
		results={}
		lamdadict={}
		for lamda in [1,5,10,15,20]:
			args['lamda']=lamda
			s1=ql_scorer(dictionary,pdf,pwC,args)
			s1.train()
			rl1=s1.rank(testscore)
			mrr, ndcg5, ndcg10 = compute_scores(rl1)
			lamdadict[lamda]=(mrr,ndcg5,ndcg10)
		args['lamda']=10
		results['lamda']=lamdadict
		coeffdict={}
		coeffs=[5,5,5]
		for j in range(3):
			for i in [1,3,5,7,9]:
				args['coeffs']=copy.deepcopy(coeffs)
				args['coeffs'][j]=i
				s=sum(args['coeffs'])
				l=[str(x) for x in args['coeffs']]
				args['coeffs']=[x/s for x in args['coeffs']]
				s1=ql_scorer(dictionary,pdf,pwC,args)
				s1.train()
				rl1=s1.rank(testscore)
				mrr, ndcg5, ndcg10 = compute_scores(rl1)
				coeffdict[l[0]+"-"+l[1]+"-"+l[2]]=(mrr,ndcg5,ndcg10)
		results['coeff']=coeffdict
		with open("queryGeneration_results.txt",'w') as f:
			json.dump(results,f)


	if sys.argv[1]=='dg':
		results={}
		for method in ['rsj','tfidf']:
			s1=dg_scorer(method,tfidf,dictionary,pdf,args,avgdl)
			s1.train()
			rl1=s1.rank(testscore)
			mrr, ndcg5, ndcg10 = compute_scores(rl1)
			results[method]=(mrr,ndcg5,ndcg10)
		bm25results={}
		kdict,bdict={},{}
		for k in [0.8,1.0,1.2,1.4,1.6]:
			args['k']=k
			s1=dg_scorer('bm25',tfidf,dictionary,pdf,args,avgdl)
			s1.train()
			rl1=s1.rank(testscore)
			mrr, ndcg5, ndcg10 = compute_scores(rl1)
			kdict[k]=(mrr,ndcg5,ndcg10)
		args['k']=1.2
		for b in [0.5,0.6,0.75,0.9,1]:
			args['b']=b
			s1=dg_scorer('bm25',tfidf,dictionary,pdf,args,avgdl)
			s1.train()
			rl1=s1.rank(testscore)
			mrr, ndcg5, ndcg10 = compute_scores(rl1)
			bdict[b]=(mrr,ndcg5,ndcg10)
		bm25results['k']=kdict
		bm25results['b']=bdict
		results['bm25']=bm25results
		with open("docGeneration_results.txt",'w') as f:
			json.dump(results,f)

			


if __name__ == '__main__':
	args={}
	args['datadir']="data/linkSO"
	args['qtype']='python'
	args['lamda']=10
	args['beta']=100
	args['fields']=['qHeader','qDescription','topVotedAnswer']
	args['coeffs']=[0.33,0.33,0.33]
	args['k']=1.2
	args['b']=0.75
	main(args)