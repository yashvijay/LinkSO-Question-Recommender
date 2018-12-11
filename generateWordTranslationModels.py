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


def generateModels(qtype):
	# dictionary, pwC, pdf = prepare_corpus("data/linkSO",recompute=False)
	datadir = "data/linkSO"
	all_questions = pd.read_csv(join(datadir, "linkso/topublish/" + qtype + "/" + qtype + "_qid2all.txt"), sep='\t', \
								names=['qID', 'qHeader', 'qDescription', 'topVotedAnswer', 'type'])
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
		bitext_qHqD_qHqD.append(AlignedSent(q1header + q1desc, q2header + q2desc))
		loop_counter += 1

	# Model 1
	print("Training Model1 QH QH..")
	start = time.time()
	ibmQH = IBMModel1(bitext_qH_qH, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQH_Model1_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQH, fout)

	print("Training Model1 QD QD..")
	start = time.time()
	ibmQD = IBMModel1(bitext_qD_qD, 50)
	print("Model QD QD trained.. In", time.time() - start, " seconds..")
	with open('modelQDQD_Model1_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQD, fout)

	print("Training Model1 QHQD QHQD..")
	start = time.time()
	ibmQHQD = IBMModel1(bitext_qHqD_qHqD, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQD_Model1_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQHQD, fout)

	print(round(ibmQH.translation_table['html']['web'], 10))
	print(round(ibmQD.translation_table['html']['web'], 10))
	print(round(ibmQHQD.translation_table['html']['web'], 10))

	# Model 2
	print("Training Model2 QH QH..")
	start = time.time()
	ibmQH = IBMModel2(bitext_qH_qH, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQH_Model2_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQH, fout)

	print("Training Model2 QD QD..")
	start = time.time()
	ibmQD = IBMModel2(bitext_qD_qD, 50)
	print("Model QD QD trained.. In", time.time() - start, " seconds..")
	with open('modelQDQD_Model2_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQD, fout)

	print("Training Model2 QHQD QHQD..")
	start = time.time()
	ibmQHQD = IBMModel2(bitext_qHqD_qHqD, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQD_Model2_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQHQD, fout)

	print(round(ibmQH.translation_table['html']['web'], 10))
	print(round(ibmQD.translation_table['html']['web'], 10))
	print(round(ibmQHQD.translation_table['html']['web'], 10))

	# Model 3
	print("Training Model3 QH QH..")
	start = time.time()
	ibmQH = IBMModel3(bitext_qH_qH, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQH_Model3_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQH, fout)

	print("Training Model3 QD QD..")
	start = time.time()
	ibmQD = IBMModel3(bitext_qD_qD, 50)
	print("Model QD QD trained.. In", time.time() - start, " seconds..")
	with open('modelQDQD_Model3_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQD, fout)

	print("Training Model3 QHQD QHQD..")
	start = time.time()
	ibmQHQD = IBMModel3(bitext_qHqD_qHqD, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQD_Model3_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQHQD, fout)

	print(round(ibmQH.translation_table['html']['web'], 10))
	print(round(ibmQD.translation_table['html']['web'], 10))
	print(round(ibmQHQD.translation_table['html']['web'], 10))

	# Model 4
	print("Training Model4 QH QH..")
	start = time.time()
	ibmQH = IBMModel4(bitext_qH_qH, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQH_Model4_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQH, fout)

	print("Training Model4 QD QD..")
	start = time.time()
	ibmQD = IBMModel4(bitext_qD_qD, 50)
	print("Model QD QD trained.. In", time.time() - start, " seconds..")
	with open('modelQDQD_Model4_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQD, fout)

	print("Training Model4 QHQD QHQD..")
	start = time.time()
	ibmQHQD = IBMModel4(bitext_qHqD_qHqD, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQD_Model4_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQHQD, fout)

	print(round(ibmQH.translation_table['html']['web'], 10))
	print(round(ibmQD.translation_table['html']['web'], 10))
	print(round(ibmQHQD.translation_table['html']['web'], 10))

	# Model5
	print("Training Model5 QH QH..")
	start = time.time()
	ibmQH = IBMModel5(bitext_qH_qH, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQH_Model5_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQH, fout)

	print("Training Model5 QD QD..")
	start = time.time()
	ibmQD = IBMModel5(bitext_qD_qD, 50)
	print("Model QD QD trained.. In", time.time() - start, " seconds..")
	with open('modelQDQD_Model5_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQD, fout)

	print("Training Model5 QHQD QHQD..")
	start = time.time()
	ibmQHQD = IBMModel5(bitext_qHqD_qHqD, 50)
	print("Model QH QH trained.. In", time.time() - start, " seconds..")
	with open('modelQHQD_Model5_' + qtype + '.pk', 'wb') as fout:
		pickle.dump(ibmQHQD, fout)

	print(round(ibmQH.translation_table['html']['web'], 10))
	print(round(ibmQD.translation_table['html']['web'], 10))
	print(round(ibmQHQD.translation_table['html']['web'], 10))


def load_model_test(model_name):
	with open(model_name, 'rb') as fin:
		ibm1 = pickle.load(fin)
	print("Loaded IBM Model:", model_name)
	print(round(ibm1.translation_table['html']['web'], 10))


if __name__ == '__main__':
	generateModels('python')
	generateModels('javascript')
	generateModels('java')
