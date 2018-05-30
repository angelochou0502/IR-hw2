import numpy as np
from collections import Counter
import os
import json
import csv
import re
import string

#get stopword list
stopwords = []
with open('data/stopwords_long.txt' , 'r') as f:
	for word in f.read().split('\n'):
		stopwords.append(word)
#define depuntuation translator
translator = str.maketrans('' , '' , string.punctuation)

def preprocess(text):
	word_list = []
	text = re.sub(r"\\n|\\T" , " " , text , flags=re.IGNORECASE)
	for word in text.translate(translator).split():
		word = word.lower()
		if word not in stopwords and not re.search(r'\d' , word):
			word_list.append(word)
	return word_list

def read_csv(file_name , value_index):
	dict_tmp = {}
	with open(file_name , 'r') as f:
		rows = csv.reader(f)
		next(rows, None)
		for row in rows:
			dict_tmp[str(row[0])] = preprocess(row[value_index])
	return dict_tmp

def write_csv( answer , file_name ):
	with open( file_name , 'w') as f:
		s = csv.writer(f)
		s.writerow(['doc_id','class_id'])
		for i,ans in enumerate(answer):
			s.writerow([i , int(answer[i])])

def log_likelihood(doc_word_matrix , topic_word_prob , doc_topic_prob):
	return np.sum((np.multiply(np.log(np.dot(doc_topic_prob , topic_word_prob)) , doc_word_matrix)))

class Group():
	def __init__(self , group_path):
		self.group_path = group_path
		self.group_dict =  read_csv(group_path , value_index = 2)
	def add_dictionary(self , dictionary_path):
		'''
		add words in the same plsa topic
		'''
		my_dict = read_csv(dictionary_path , value_index = 1)
		for class_word in self.group_dict.values():
			for dic_list in my_dict.values():
				if(class_word[0] in dic_list):
					class_word += dic_list
	#after creating doc_topic_prob and topic_word_prob
	def predict(self, vocabulary , out_file):
		doc_topic_prob = np.load('./data/doc_topic_prob.npy')
		topic_word_prob = np.load('./data/topic_word_prob.npy')
		numbers_of_documents = doc_topic_prob.shape[0]
		numbers_of_topic = doc_topic_prob.shape[0]
		numbers_of_vocabulary = topic_word_prob.shape[1]
		doc_word_matrix = np.load('./data/doc_word_matrix_%d.npy' %(numbers_of_vocabulary))

		doc_word_prob = np.zeros((numbers_of_documents , numbers_of_topic))
		answer = np.zeros(numbers_of_documents)

		for class_index in self.group_dict:
			word_num = len(self.group_dict[class_index])
			for word in self.group_dict[class_index]:
				word_index = vocabulary.index(word)
				doc_word_prob[:, int(class_index)] += np.dot(doc_topic_prob , topic_word_prob[:, word_index])
			doc_word_prob[:, int(class_index)] /= word_num
		for i,doc in enumerate(doc_word_prob):
			answer[i] = np.argmax(doc)

		#if the word appear in the text -> the same type
		for g_i,key in enumerate(self.group_dict):
			index = vocabulary.index(self.group_dict[key][0])
			for i , d in enumerate(doc_word_matrix):
				if(d[index] > 0):
					answer[i] = g_i
		#write answer
		write_csv(answer , out_file)

class Document():
	def __init__(self , doc_path):
		self.doc_path = doc_path
		self.doc_dic = {} 
		self.vocabulary = [] # word that will be calculated

	def get_doc_dict(self):
		#check if json file already exists
		if(os.path.isfile('./data/doc_dict.json')):
			print('loading document dictionary...')
			with open('./data/doc_dict.json' , 'r') as f:
				self.doc_dict = json.load(f)
		else:
			print('creating document dictionary...')
			self.doc_dict = read_csv(self.doc_dict , value_index = 1)
			with open('./data/doc_dict.json' , 'w') as f:
				json.dump(doc_dict , f)
	def get_vocabulary(self, min_count):
		#get the total word list and only care for word count > 200
		total_list = []
		for word_list in self.doc_dict.values():
			total_list += word_list
		word_count = Counter(total_list)
		for [word , count] in word_count.items():
			if(count > min_count):
				self.vocabulary.append(word)
	def plsa(self , number_of_topic , max_iter):
		numbers_of_documents = len(self.doc_dict)
		numbers_of_vocabulary = len(self.vocabulary)

		#construc doc_word_matrix
		doc_word_matrix = np.zeros((numbers_of_documents , numbers_of_vocabulary))
		if(os.path.isfile('./data/doc_word_matrix_%d.npy' %(numbers_of_vocabulary))):
			print('loading doc_word_matrix...')
			doc_word_matrix = np.load('./data/doc_word_matrix_%d.npy' %(numbers_of_vocabulary))
		else:
			print('creating word doc matrix...')
			for d_index, key in enumerate(self.doc_dict):
				word_count = np.zeros(numbers_of_vocabulary)
				for word in self.doc_dict[key]:
					if word in self.vocabulary:
						w_index = self.vocabulary.index(word)
						word_count[w_index] += 1
				doc_word_matrix[d_index] = word_count
			np.save('./data/doc_word_matrix_%d.npy' %(numbers_of_vocabulary) , doc_word_matrix)

		#initial the probablity of p(z | d) and p (w | z) and normalize
		print("Initializatig...")
		self.doc_topic_prob = np.random.random(((numbers_of_documents , number_of_topic)))
		for d in range(numbers_of_documents):
			self.doc_topic_prob[d] = self.doc_topic_prob[d] / np.sum(self.doc_topic_prob[d])
		self.topic_word_prob = np.random.random((number_of_topic , numbers_of_vocabulary))
		for t in range(number_of_topic):
			self.topic_word_prob[t] = self.topic_word_prob[t] / np.sum(self.topic_word_prob[t])
		print("Starting EM...")
		for i in range(max_iter):
			print("Iteration #%d" %(i+1))
			print("E step...")
			topic_prob = np.zeros((numbers_of_documents , numbers_of_vocabulary , number_of_topic) ,  dtype = np.float64)
			for d in range(numbers_of_documents):
				topic_prob[d] = np.multiply(self.doc_topic_prob[d , :] , self.topic_word_prob.transpose())
				#divide_row -> deal with divided by zero
				divide_row = np.ones(numbers_of_vocabulary)
				greater_zero_index = np.where(np.sum(topic_prob[d] , 1) > 0)
				divide_row[greater_zero_index] = np.sum(topic_prob[d] , 1)[greater_zero_index]
				topic_prob[d] = (topic_prob[d].transpose() / divide_row).transpose()
				#topic_prob[d] = (topic_prob[d].transpose() / np.sum(topic_prob[d] , 1)).transpose()

			print("M step...")
			#update self.doc_topic_prob (P(z | d))
			for d in range(numbers_of_documents):
				self.doc_topic_prob[d] = np.dot(doc_word_matrix[d] , topic_prob[d])
				if(np.sum(doc_word_matrix[d]) > 0):
					self.doc_topic_prob[d] = self.doc_topic_prob[d] / np.sum(doc_word_matrix[d])
			#update self.topic_word_prob (p(w | z))
			for t in range(number_of_topic):
				self.topic_word_prob[t] = np.einsum('ij , ij->i' , doc_word_matrix.transpose() , topic_prob[:,:,t].transpose())
				if(np.sum(self.topic_word_prob[t]) > 0):
					self.topic_word_prob[t] = self.topic_word_prob[t] / np.sum(self.topic_word_prob[t])
			print("log likelihood: %f" %(log_likelihood(doc_word_matrix , self.topic_word_prob , self.doc_topic_prob)))

		np.save('./data/topic_word_prob.npy' , self.topic_word_prob)
		np.save('./data/doc_topic_prob.npy' , self.doc_topic_prob)



