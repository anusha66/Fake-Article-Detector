import preprocessData as pd
import numpy as np 

import nltk

def count(squence, item):
    cnt = 0
    for i in squence:
        if i == item:
            cnt += 1

    return cnt
def compute_tred(articles):
	stats = []
	loss_articles = [] #mean,median,minimun,maximum
	sparse = []
	for art in articles:
		## Compute Vocab and Article Sentence Length
		n_sent = len(art.allSentences)
		vocab = {}
		
		for sent in art.allSentences:
			words = nltk.word_tokenize(sent.string)
			for w in words:
				if w not in vocab:
					vocab[w] = 1
				else:
					vocab[w] += 1
		n_vocab = len(vocab.keys())
		d = np.zeros([n_vocab,n_sent])

		#populating the d matrix
		s_index = -1
		for sent in art.allSentences:
			s_index += 1
			words = nltk.word_tokenize(sent.string)
			for w in vocab.keys():
				w_index = hash(w)%(n_vocab)
				d[w_index][s_index] = count(words,w)
		#print d
		
		#print "~~~~~~~~~~"

		#Constructed sentence-sentence matrix
		a = np.dot(np.transpose(d),d)
		#print a.shape
		
		u,s,v = np.linalg.svd(a, full_matrices=True)
		a = np.dot(u,s)
		a = np.dot(a,v)
		#print u.shape
		#print u.shape,a.shape()
		#print "---------------"

		## Top 10 eigen values from S
		N = 10
		s_temp = s.argsort()[-N:][::-1]
		s_dash = np.zeros(s.shape)
		for i in range(len(s_temp)):
			s_dash[i] = s[s_temp[i]]


		#s_dash is S prime
		#print s_dash.shape,s.shape
		#print "---------------------"
		#print s_dash

		a_prime = np.dot(u,s_dash)
		a_prime = np.dot(a_prime,v)
		#print a_prime

		#### Calculating Information Loss #########
		loss = np.square(np.linalg.norm(np.subtract(a,a_prime)))
		
		loss_articles.append(loss)
		#print loss

		#Stats for a_prime
		mean = np.mean(a_prime)
		# print mean
		median = np.median(a_prime)
		maximum = np.max(a_prime)
		minimun = np.min(a_prime)
		stats.append((mean,median,maximum,minimun))

		#Sparsity of A_PRIME
		sparse_score = np.count_nonzero(a_prime)/float(n_vocab*n_sent)
		#print sparse_score
		sparse.append(sparse_score)


	
	return loss_articles,stats,sparse
	

def normalize(element):
	mean = np.mean(element)
	deviation = np.std(element)
	#for i in range(len(element)):
	#	element[i] = (element[i] - mean)/float(deviation)
	#return element







def main():
	    trainArticles = pd.preprocessData("trainingSet.dat", "trainingSetLabels.dat")

	    #testArticles = pd.preprocessData("developmentSet.dat", "developmentSetLabels.dat")

	    loss,stats,sparse = compute_tred(trainArticles)
	    for ele in loss:
	    	print loss
	    



if __name__ == "__main__":
    main()