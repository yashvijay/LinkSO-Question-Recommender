import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import logging
import json
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from nltk.translate import AlignedSent
from nltk.translate import Alignment
from nltk.translate import IBMModel, IBMModel1, IBMModel2, IBMModel3, IBMModel4, IBMModel5
from nltk.translate.ibm_model import Counts
import time
import dill as pickle

from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

question_id = 'qID'
question_header = 'qHeader'
question_description = 'qDescription'
top_answer = 'topVotedAnswer'
type = 'type'
question_id1 = 'qID1'
question_id2 = 'qID2'
score = 'score'
label = 'label'


# def generate_similar_tuples(file):
# 	result = []
# 	for each_line in file:
# 		print(each_line)
# 		# if each_line[label] == '1':
# 		# 	result.append((each_line[question_id1], each_line[question_id2]))
# 	return result

def main():
	# dictionary, pwC, pdf = prepare_corpus("data/linkSO",recompute=False)
	datadir = "data/linkSO"
	qtype = "python"
	all_questions = pd.read_csv(join(datadir, "linkso/topublish/" + qtype + "/" + qtype + "_qid2all.txt"), sep='\t', \
	 							names=['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
	# print("QID:", all_questions[question_id])
	# print("QHead:", all_questions[question_header])
	# print("QDesc:", all_questions[question_description])
	# print("Answer:", all_questions[top_answer])
	# print("Type:", all_questions[type])
	# dictionary, pwC, pdf = prepare_corpus("data/linkSO", recompute=False)
	# print("Dictionary:", dictionary)
	# print("pwC:", pwC)
	# # print("pdf", pdf)
	similar_docs_file = pd.read_csv(join(datadir, "linkso/topublish/" + qtype + "/" + qtype + "_cosidf.txt"), sep='\t', \
									names=['qID1', 'qID2', 'score', 'label'], skiprows=1)
	filtered_rows = similar_docs_file[similar_docs_file[label] == 1]
	filtered_columns = filtered_rows.filter(items=[question_id1, question_id2])
	bitext_qH_qH = []
	bitext_qD_qD = []
	bitext_qHqD_qHqD = []
	loop_counter = 0
	for each_row in filtered_columns.itertuples():
		q1ID = each_row[1]
		q2ID = each_row[2]
		q1_row = all_questions.loc[all_questions[question_id] == q1ID]
		q1header = str(q1_row[question_header].values[0]).split()
		q1desc = str(q1_row[question_description].values[0]).split()
		q1ans = str(q1_row[top_answer].values[0]).split()
		q2_row = all_questions.loc[all_questions[question_id] == q2ID]
		q2header = str(q2_row[question_header].values[0]).split()
		q2desc = str(q2_row[question_description].values[0]).split()
		q2ans = str(q2_row[top_answer].values[0]).split()
		print("\nQ1 Header:", q1header)
		print("Q1 Desc:", q1desc)
		print("Q1 Answer:", q1ans)
		print("Q2:", q2header)
		print("Q2 Desc:", q2desc)
		print("Q2 Answer:", q2ans)
		bitext_qH_qH.append(AlignedSent(q1header, q2header))
		bitext_qD_qD.append(AlignedSent(q1desc, q2desc))
		bitext_qHqD_qHqD.append(AlignedSent(q1header+q1desc, q2header+q2desc))
		loop_counter += 1

	print("Training Model QH QH..")
	start = time.time()
	ibmQH = IBMModel1(bitext_qH_qH, 50)
	print("Model QH QH trained.. In", time.time()-start , " seconds..")
	with open('modelQHQH_Model1_python.pk', 'wb') as fout:
		pickle.dump(ibmQH, fout)

	print("Training Model QD QD..")
	start = time.time()
	ibmQD = IBMModel1(bitext_qD_qD, 50)
	print("Model QD QD trained.. In", time.time()-start , " seconds..")
	with open('modelQDQD_Model1_python.pk', 'wb') as fout:
		pickle.dump(ibmQD, fout)

	print("Training Model QHQD QHQD..")
	start = time.time()
	ibmQHQD = IBMModel1(bitext_qHqD_qHqD, 50)
	print("Model QH QH trained.. In", time.time()-start , " seconds..")
	with open('modelQHQD_Model1_python.pk', 'wb') as fout:
		pickle.dump(ibmQHQD, fout)

	print(round(ibmQH.translation_table['html']['web'], 10))
	print(round(ibmQD.translation_table['html']['web'], 10))
	print(round(ibmQHQD.translation_table['html']['web'], 10))



def prepare_corpus(datadir, qtype='python', recompute=False):
	pdf = pd.read_csv(join(datadir, "linkso/topublish/" + qtype + "/" + qtype + "_qid2all.txt"), sep='\t',
					  names=['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
	pdf['type'] = qtype
	if isfile(join(datadir, qtype + "_qa.dict")) and (recompute == False):
		dictionary = corpora.Dictionary.load(join(datadir, qtype + "_qa.dict"))
		with open(join(datadir, qtype + "_pwC.json"), 'r') as f:
			pwC = json.load(f)
		# corpus = MyCorpus(join(datadir,qtype+"_corpus.txt"),dictionary)
	else:

		pdf.loc[:, 'text'] = pdf['qHeader'].astype(str) + pdf['qDescription'].astype(str) + pdf[
			'topVotedAnswer'].astype(str)

		documents = pdf['text']
		frequency = defaultdict(int)
		numtokens = 0
		for d in documents:
			for token in d.split():
				frequency[token] += 1
				numtokens += 1

		bowdocs = [d.split() for d in documents]
		dictionary = corpora.Dictionary(bowdocs)
		dictionary.save(join(datadir, qtype + "_qa.dict"))

		# corpus = MyCorpus(join(datadir,qtype+"_corpus.txt"),dictionary,documents=documents)
		pwC = {}
		for token in frequency:
			pwC[dictionary.token2id[token]] = frequency[token] / numtokens
		with open(join(datadir, qtype + "_pwC.json"), 'w') as f:
			json.dump(pwC, f)

	return dictionary, pwC, pdf


def load_model_test(model_name):
	with open(model_name, 'rb') as fin:
		ibm1 = pickle.load(fin)
	print("Loaded IBM Model:", model_name)
	print(round(ibm1.translation_table['html']['web'], 10))

if __name__ == '__main__':
	main()
	# load_model_test('modelQHQH_toy.pk')
	# load_model_test('modelQDQD_toy.pk')
	# load_model_test('modelQHQD_toy.pk')
	# datadir = "data/linkSO"
	# qtype = "python"
	# all_questions = pd.read_csv(join(datadir, "linkso/topublish/" + qtype + "/" + qtype + "_qid2all.txt"), sep='\t',
	# 							names=['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
	# similar_docs_file = pd.read_csv(join(datadir, "linkso/topublish/" + qtype + "/" + qtype + "_cosidf.txt"), sep='\t',
	# 								names=['qID1', 'qID2', 'score', 'label'], skiprows=1)
	# filtered_rows = similar_docs_file[similar_docs_file[label] == 1]
	# filtered_columns = filtered_rows.filter(items=[question_id1, question_id2])
	# save_models()
