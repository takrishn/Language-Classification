import os

import nltk 
from nltk import word_tokenize
from nltk.probability import FreqDist #imported to avoid error

import json
import sklearn
from sklearn.feature_extraction.text import * 
from sklearn.model_selection import train_test_split 

from sklearn import linear_model 
from sklearn import metrics 

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from collections import defaultdict
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def run_bayes():
	classification_mapping = {
		"English": 0, 
		"Spanish": 1, 
		"German": 2,
		"Italian": 3,
		"French": 4,
		"Simlish": 5,
		"Tolkien Elvish": 6
	}

	df = pd.read_csv('language_dataset.csv')

	# Remove Missing values
	# Add a column encoding the product as an integer
	col = ['Word', 'Language']
	df = df[col]
	df = df[pd.notnull(df['Language'])]
	df.columns = ['Word', 'Language']
	df['category_id'] = df['Word'].factorize()[0]
	category_id_df = df[['Word', 'category_id']].drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_df.values)
	id_to_category = dict(category_id_df[['category_id', 'Word']].values)
	df.head()

	## plot to flex
	# fig = plt.figure(figsize=(8,6))
	# df.groupby('Product').Consumer_Complaint.count().plot.bar(ylim=0)
	# plt.show()

	train, test = train_test_split(df, random_state = 0, test_size=.25)

	tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 1))
	features = tfidf.fit_transform(train.Word).toarray()
	labels = train.category_id
	features.shape

	# Use 'sklearn.feature_selection.chi2' to find the terms that are the most correlated with each of the products
	N = 2
	for Language, category_id in sorted(category_to_id.items()):
		features_chi2 = chi2(features, labels == category_id)
		indices = np.argsort(features_chi2[0])
		feature_names = np.array(tfidf.get_feature_names())[indices]
		unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
		bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
		#   print("--> '{}':".format(Language))
		#   print("  . Most Correlated Unigrams are :\n. {}".format('\n. '.join(unigrams[-N:])))
		#   print("  . Most Correlated Bigrams are :\n. {}".format('\n. '.join(bigrams[-N:])))

	X_train = train['Word']
	Y_train = train['Language']
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(X_train)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	clf = MultinomialNB().fit(X_train_tfidf, Y_train)


	correct_predictions = 0
	for ind in test.index:
		prediction = clf.predict(count_vect.transform([test['Word'][ind]]))[0]

		correct = 'O' if prediction == test['Language'][ind] else 'X'
		print('word: {}, prediction: {}, actual: {}, Correct: {}'.format(test['Word'][ind], prediction, test['Language'][ind], correct))

		if correct == 'O':  
			correct_predictions += 1
	print('Accuracy: {}%'.format(correct_predictions / len(test.index) * 100))

	print(clf.predict(count_vect.transform(["event"])))



def run_logistic():
	# ---------------------------------------------------------------------------------------
	# PROBLEM 4
	# Create a bag of words (BOW) representation from text documents, using the Vectorizer function in scikit-learn
	#
	# The inputs are 
	#  - a filename (you will use "yelp_reviews.json") containing the reviews in JSON format 
	#  - the min_pos and max_neg parameters
	#  - we label all reviews with scores > min_pos = 4 as "1"  
	#  - we label all reviews with scores < max_neg = 2 as "0" 
	#  - this creates a simple set of labels for binary classification, ignoring the neutral (score = 3) reviews
	# 
	#  The function extracts the text and scores for each review from the JSON data
	#  It then tokenizes and creates a sparse bag-of-words array using scikit-learn vectorizer function
	#  The number of rows in the array is the number of reviews with scores <=2 or >=4
	#  The number of columns in the array is the number of terms in the vocabulary
	#
	#  NOTE: 
	#  - please read the scikit-learn tutorial on text feature extraction before you start this problem:
	#     https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction  
	#  - in this function we will use scikit-learn tokenization (rather than NLTK)
	# ---------------------------------------------------------------------------------------

	def create_bow_from_json(f):
		def parseCSV(f):
			data = pd.read_csv(f)
			return data

		words = [] #all the words
		Y = [] #list of classifications

		classification_mapping = {
			"English": 0, 
			"Spanish": 1, 
			"German": 2,
			"Italian": 3,
			"French": 4,
			"Simlish": 5,
			"Tolkien Elvish": 6
		}
		
		csv_data = parseCSV(f).to_dict('split')['data']
		category_lines = defaultdict(list)
		for entry in csv_data[1:]: #needed to skip the csv header
			words.append(entry[0])
			Y.append(classification_mapping[entry[1]])

		# create an instance of a CountVectorizer, using 
		# (1) Already removed stopwords during data collection step
		
		# min_df=0.01 means to ignore all terms that appear in less than 1% of all docs
		vectorizer = CountVectorizer()
		
		# create a sparse BOW array from 'text' using vectorizer  
		X = vectorizer.fit_transform(words)
		print('Data shape: ', X.shape)
		
		return X, Y, vectorizer
			
	# ---------------------------------------------------------------------------------------
	# PROBLEM 5
	#  Separate an X,Y dataset (X=features, Y=labels) into training and test subsets
	#  Build a logistic classifier on the training subset
	#  Evaluate performance on the test subset  
	#
	#  NOTE: before starting this problem please read the scikit-learn documentation on logistic classifiers:
	#		https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
	# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
	# ---------------------------------------------------------------------------------------		 
	def logistic_classification(X, Y, test_fraction): 
		# should add comments defining what the inputs are what the function does
		"""
			X = a sparse matrix of the words in the corpus (vocabulary), where 
			each column represents a different word, and each row represents a document;
			each column's number indicates the num of occurances of that word in the document
			Y = the binary representation of the review (0 means negatie, 1 means positive)
			test_fraction = how much of the data we are splitting to use for testing (with the
			other fraction used for training)
		"""

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
		#  set the state of the random number generator so that we get the same results across runs when testing our code
		
		print('Number of training examples: ', X_train.shape[0])
		print('Number of testing examples: ', X_test.shape[0])   
		print('Vocabulary size: ', X_train.shape[1]) 
	

		# Specify the logistic classifier model with an l2 penalty for regularization and with fit_intercept turned on
		classifier = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, multi_class='multinomial', solver='lbfgs') 
		
		# Train a logistic regression classifier and evaluate accuracy on the training data
		print('\nTraining a model with', X_train.shape[0], 'examples.....')

		# .... fit the classification model.....
		classifier.fit(X_train, Y_train) # fit using training data

		train_predictions = classifier.predict(X_train)	  
		train_accuracy = metrics.accuracy_score(train_predictions, Y_train)
		print('\nTraining:')
		print(' accuracy:',format( 100*train_accuracy , '.2f') ) 

		# Compute and print accuracy and AUC on the test data
		print('\nTesting: ')
		test_predictions = classifier.predict(X_test)	 
		test_accuracy = metrics.accuracy_score(test_predictions, Y_test)
		print(' accuracy:', format( 100*test_accuracy , '.2f') )
		
		# probabilities for class 1
		class_probabilities = classifier.predict_proba(X_test)[:,1]
		# test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities, average='micro')
		# print(' AUC value:', format( 100*test_auc_score , '.2f') )
		
		return(classifier) 
		
	# ---------------------------------------------------------------------------------------
	# PROBLEM 6
	#   Takes as input
	#     (1) a scikit-learn trained logistic regression classifier (e.g., trained in Problem 5) 
	#     (2) a scikit-learn vectorizer object that produced the BOW features for the classifier
	#   and prints out and returns
	#   - the K terms in the vocabulary tokens with the largest positive weights  
	#   - the K terms in the vocabulary with the largest negative weights 
	#
	# To write this code you will need to use the get_params() method for the logistic regression model 
	# in scikit-learn, and you will also need to access the terms (strings) in the vocabulary in the 
	# Vectorizer object and match them up to the corresponding weights in the logistic classifier model.
	# ---------------------------------------------------------------------------------------				

	def most_significant_terms(classifier, vectorizer, K):	
		# You can write this function in whatever way you want
		
		# ...after you find the relevant weights and terms....     
		# ....cycle through the positive weights, in the order of largest weight first and print out
		# K lines where each line contains 
		# (a) the term corresponding to the weight (a string)
		# (b) the weight value itself (a scalar printed to 3 decimal places)
		
		# e.g., for w in topK_pos_weights:
		#      	...
		#      	print( ....
		#      	
		# Same for negative weights, most negative values first
		#      for w in topK_neg_weights:
		#      	...
		#      	print( ....

		# get all the terms of the corpus
		vocabulary = vectorizer.get_feature_names()
	
	
	# get the weights of the terms; [0] is there to extract the inner list that
		# actually holds the weight values
		weights = classifier.coef_[0]

		# get the top K positive indices ([::-1] to sort in descending order)
		topK_pos_indices = weights.argsort()[-K:][::-1]

		# get the top K negative indices (largest negative value first)
		topK_neg_indices = weights.argsort()[:K]

		topK_pos_weights = []
		topK_neg_weights = []

		topK_pos_terms = []
		topK_neg_terms = []

		#print('Top K pos indices: ', topK_pos_indices)
		#print('Top K neg indices: ', topK_neg_indices)

		# print the top K terms and weights, and append them to respective lists
		print('\nTop', K, 'Positive Terms:')
		for i in topK_pos_indices:
			print('Term: ', vocabulary[i], '\t\tWeight: ', format(weights[i] , '.3f'))
			topK_pos_weights.append(weights[i])
			topK_pos_terms.append(vocabulary[i])

		print('\n Top', K, 'Negative Terms:')
		for i in topK_neg_indices:
			print('Term: ', vocabulary[i], '\t\tWeight: ', format(weights[i] , '.3f'))
			topK_neg_weights.append(weights[i])
			topK_neg_terms.append(vocabulary[i])
		
		return(topK_pos_weights, topK_neg_weights, topK_pos_terms, topK_neg_terms)
			
	print('\n-----PROBLEM 4-----')
	# read in our datafile and tokenize the text for each catagory
	X, Y , vectorizer_BOW = create_bow_from_json("language_dataset.csv")  

	print('\n-----PROBLEM 5-----')
	# run a logistic classifier on the reviews, specifying the fraction to be used for testing  
	test_fraction = 0.5
	logistic_classifier = logistic_classification(X, Y,test_fraction)  

	#test_fraction = 0.8
	#logistic_classifier = logistic_classification(X, Y,test_fraction)  
	
	print('\n-----PROBLEM 6-----')
	# print out and return the most significant positive and negative weights (and associated terms) 
	most_significant_terms(logistic_classifier, vectorizer_BOW, K=10)

if __name__ == '__main__':
	print('MULTINOMIAL NAIVE BAYES!!!')
	run_bayes()
	print('LOGISTIC REGRESSION!!!')
	run_logistic()