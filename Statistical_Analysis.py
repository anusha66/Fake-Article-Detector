import nltk, gensim, numpy as np, sklearn
from gensim.models import Word2Vec
from Load_Data import Load_Data
from sklearn.manifold import TSNE
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.naive_bayes import GaussianNB as GNB
from featureExtractor import extractFourGram
from sklearn.linear_model import LogisticRegression
import pickle

import matplotlib
matplotlib.use('TkAgg')
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools


class Stat_Analysis():
	def __init__(self, true_sentences, fake_sentences, labels):
		# self.train_data = train_data
		self.true_sentences = true_sentences
		self.fake_sentences = fake_sentences
		self.labels = labels

	def pos_load_features(self):
		train_pos = pickle.load(open("article_gram_score.pkl","rb"))

		dev_pos = pickle.load(open("devarticle_gram_score.pkl","rb"))

		return train_pos, dev_pos

	def remove_stop_words(self,sentences):
		stop_words = get_stop_words('en')
		for x in sentences:
			for i in range(len(x)):
				if x[i] in stop_words:
					x.pop(i)

	def get_dev_labels(self):
		y_dev = []
		with open("developmentSetLabels.dat","r") as dev_file:
			for line in dev_file:
				y_dev.append(int(line))

		return y_dev


	def word2vec(self):
		'''
		Function to train wrod2vec embeddings and plot embeddings in vector space
		'''
		true_sentences = [x.split(" ") for x in self.true_sentences]
		self.remove_stop_words(true_sentences)
		bigram_transformer = gensim.models.Phrases(true_sentences)
		true_model = Word2Vec(bigram_transformer[true_sentences], min_count=100)
		
		fake_sentences = [x.split(" ") for x in self.fake_sentences]
		self.remove_stop_words(fake_sentences)
		bigram_transformer = gensim.models.Phrases(fake_sentences)
		fake_model = Word2Vec(bigram_transformer[fake_sentences], min_count=100)

		self.plot_tsne(true_model, fake_model)

	def make_feature_graph(self, feature_list, labels_filename="trainingSetLabels.dat"):
		'''
			Function to plot 2 graphs:
				1. Decision Boundaries: Takes atmost 2 features for every sample and plots decision boundaries defined by 5 classifiers: 
					['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM', 'AdaBoost']
				2. Scatter Plot: Plots the values of each data point on a Scatter plot to visualise how separable they seem.
								 This is not performed on any classifier. For manual evaluation only. 

			Parametrs:
				feature_list: A list of lists containing the features for each sample.
				labels_filename: Path to the filename containing the labels for the training data
		'''

		y = []
		with open(labels_filename) as label_file:
			x_true_list = []
			x_fake_list = []
			for idx,label in enumerate(label_file):
				if int(label):
					y.append(1)
					x_true_list.append(feature_list[idx])
				else:
					y.append(0)
					x_fake_list.append(feature_list[idx])

		y = np.array(y)	
		X_plot = feature_list

		#---------------------------- Decision Boundary Plot -----------------------#
		if len(feature_list[0])==1 or len(feature_list[0])==2:
			print "Now plotting Decision boundary Plot. (Works best for 2 features)"
			
			gs = gridspec.GridSpec(2, 2)

			fig = plt.figure(figsize=(10,8))

			clf1 = LogisticRegression(random_state=1)
			clf2 = RFC(n_estimators=100, random_state=1)
			clf3 = GNB()
			clf4 = SVC()
			clf5 = ABC()

			labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM', 'AdaBoost']
			for clf, lab, grd in zip([clf1, clf2, clf3, clf4, clf5], labels, itertools.product([0, 1], repeat=2)):

			    clf.fit(X_plot, y)
			    ax = plt.subplot(gs[grd[0], grd[1]])
			    fig = plot_decision_regions(X=X_plot, y=y, clf=clf, legend=2)
			    plt.title(lab)

			plt.show()

		#---------------------------- Individual Scatter Plot -----------------------#
		plot_idx = 0
		if len(feature_list[0]) != 1:
			plot_idx = int(raw_input("Your list has more than 1 feature. Which feature would you like to observe? (Insert Index): "))

		print "Now plotting scatter plot of feature:"
		x_true = [feat[plot_idx] for feat in x_true_list]
		x_fake = [feat[plot_idx] for feat in x_fake_list]

		x_true = np.array(x_true)
		x_fake = np.array(x_fake)
		y_plot = np.arange(max(len(x_true),len(x_fake)))

		trace_true = go.Scatter(y=x_true, x=y_plot, mode='markers', text = "True")
		trace_fake = go.Scatter(y=x_fake, x=y_plot, mode='markers', text = "Fake")

		data = [trace_true, trace_fake]
		layout = go.Layout(showlegend=False)
		fig = go.Figure(data=data, layout=layout)
		plot_url = offline.plot(fig, filename='text-chart-basic')	    

	def article_classifier(self):

		train_pos, dev_pos = self.pos_load_features()

		rare_ttr_perplexity_4gram_features = list(extractFourGram('featureFour.txt','basic.csv'))

		X_dev = list(extractFourGram('featureFour_dev.txt','basic_dev.csv'))
		y_dev = self.get_dev_labels()

		X = rare_ttr_perplexity_4gram_features
		y = self.labels

		X.append(train_pos)
		X_dev.append(dev_pos)

		X = np.array(X).T[:,:]
		X_dev = np.array(X_dev).T[:,:]

		# self.make_feature_graph(X[:,1:3],"trainingSetLabels.dat")

		lr_clf = LogisticRegression()
		lr_clf.fit(X,y)
		lr_predicted = lr_clf.predict(X_dev)
		lr_scores = cross_val_score(lr_clf, X, y, cv=5, n_jobs = 5)
		print lr_scores,np.mean(lr_scores), np.std(lr_scores)
		# svm_predicted = cross_val_predict(lr_clf, X, y, cv=5)
		print accuracy_score(y_dev,lr_predicted)

		# SVM Parameters:
		# {'C': [0.1,1.0,10.0,100.0], 'gamma':[1.0,2.0,'auto',0.1,0.01,0.001], 'kernel':['rbf','linear']}
		svm_clf = SVC(probability=True) 
		svm_clf.fit(X,y)
		svm_predicted = svm_clf.predict(X_dev)
		svm_scores = cross_val_score(svm_clf, X, y, cv=5, n_jobs = 5)
		print svm_scores,np.mean(svm_scores), np.std(svm_scores)
		# svm_predicted = cross_val_predict(svm_clf, X, y, cv=5)
		print accuracy_score(y_dev,svm_predicted)

		# RandomForest Parameters:
		# {'n_estimators':[10,20,5,30],'criterion':['gini','entropy']}
		rf_clf = RFC() 
		rf_clf.fit(X,y)
		rf_predicted = rf_clf.predict(X_dev)
		rf_scores = cross_val_score(rf_clf, X, y, cv=5, n_jobs = 5)
		print rf_scores,np.mean(rf_scores), np.std(rf_scores)
		# rf_predicted = cross_val_predict(rf_clf, X, y, cv=5)
		print accuracy_score(y_dev,rf_predicted)

		# AdaBoost Parameters:
		# {'n_estimators':[10,20,5,30],'learning_rate':[1.0,0.1,0.01,0.001,0.05]}
		ab_clf = ABC() 
		ab_clf.fit(X,y)
		ab_predicted = ab_clf.predict(X_dev)
		ab_scores = cross_val_score(ab_clf, X, y, cv=5, n_jobs = 5)
		print ab_scores, np.mean(ab_scores), np.std(ab_scores)
		# ab_predicted = cross_val_predict(ab_clf, X, y, cv=5)
		print accuracy_score(y_dev,ab_predicted)

		# Gaussian NB Parameters:
		# {'n_estimators':[10,20,5,30],'learning_rate':[1.0,0.1,0.01,0.001,0.05]}
		nb_clf = GNB() 
		nb_clf.fit(X,y)
		nb_predicted = nb_clf.predict(X_dev)
		nb_scores = cross_val_score(nb_clf, X, y, cv=5, n_jobs = 5)
		print nb_scores, np.mean(nb_scores), np.std(nb_scores)
		# nb_predicted = cross_val_predict(nb_clf, X, y, cv=5)
		print accuracy_score(y_dev,nb_predicted)

	def plot_tsne(self,true_model, fake_model):

		true_labels = []
		true_tokens = []

		for word in true_model.wv.vocab:
			true_tokens.append(true_model[word])
			true_labels.append(word)

		tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
		new_values = tsne_model.fit_transform(true_tokens)

		x = []
		y = []
		for value in new_values:
			x.append(value[0])
			y.append(value[1])
		
		trace3 = go.Scatter(x=x, y=y, mode='markers+text', name='Lines and Text', text=true_labels, textposition='bottom')
		
		fake_labels = []
		fake_tokens = []

		for word in fake_model.wv.vocab:
			fake_tokens.append(fake_model[word])
			fake_labels.append(word)

		tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
		new_values = tsne_model.fit_transform(fake_tokens)

		x = []
		y = []
		for value in new_values:
			x.append(value[0])
			y.append(value[1])
		
		trace2 = go.Scatter(x=x, y=y, mode='markers+text', name='Lines and Text', text=fake_labels, textposition='top')

		data = [trace3, trace2]
		layout = go.Layout(showlegend=False)
		fig = go.Figure(data=data, layout=layout)
		plot_url = offline.plot(fig, filename='text-chart-basic')	        

# data_object = Load_Data("trainingSet.dat","trainingSetLabels.dat")
# true_sentences, fake_sentences, labels = data_object.return_true_fake()
# obj = Stat_Analysis(true_sentences, fake_sentences, labels)
# obj.article_classifier()


