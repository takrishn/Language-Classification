Files submitted:

CNN_notebook.html - an HTML of our jupyter notebook for running and showing the results of our RNN model
Difficult_set.cvs - similar to language_dataset_with_stopwords.cvs but with the top 50 most difficult words in each of the 5 real language include and with duplicates across languages removed
language_dataset.cvs - our first training dataset that is made up of 1000 of the most common word in each of the 5 real languages with stopwords removed plus the entire dataset for Simlish and Elvish minus Elvish stopwords
language_dataset_with_stopwords.cvs - similar to the language_dataset.csv but with stopwords included
RNN_notebook.html - an HTML of our jupyter notebook for running and showing the results of our RNN model
test_data_v1.cvs - our test data set that is created by pulling random words from each language; used against all training datasets and classifiers
test_data_v2.cvs - our test data set that is created by testing different character lengths

/src
baseline_classifier.py - naiive bayes and logistical regression classifier that we experimented with, ended up being discarded
char2vec_Edward.py - a tutorial that we followed to try to get char2vec to work on our problem but was ultimately scrapped
CNN-Difficult_set.txt - our training epoch data for Difficult_set.cvs using our CNN model
CNN-LanguageDataSet.txt - our training epoch data for Difficult_set.cvs using our CNN model
CNN-StopWords.txt - our training epoch data for Difficult_set.cvs using our CNN model
Language_Output_Accuracy.txt
languageScraper.py - all of our webscrapping code to get the language_dataset.cvs and language_dataset_with_stopwords.cvs datasets