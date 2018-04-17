import preprocessData as pd
import numpy as np 
import math
from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from nltk.corpus import stopwords
def count(squence, item):
    cnt = 0
    for i in squence:
        if i == item:
            cnt += 1

    return cnt

def idf(art,vocab):
	cnt = 0
	idf_list = {}
	for sent in art.allSentences:
		words = nltk.word_tokenize(sent.string)
		words = [word for word in words if word.lower() not in stopwords.words('english')]
		for w in vocab.keys():
			if w in words:
				if w not in idf_list:
					idf_list[w] = 1
				else:
					idf_list[w] += 1
	return idf_list


	
def compute_tred_median(articles):
	score = []
	loss_articles = [] #mean,median,minimun,maximum
	sparse = []
	no_r = no_f = 0

	#cosine_sim=[]
	no_articles = 0
	for art in articles:

		cosine_sim=[]
		
		## Compute Vocab and Article Sentence Length
		n_sent = len(art.allSentences)
		vocab = {}
		tokens = 0
		for sent in art.allSentences:

			words = nltk.word_tokenize(sent.string)
	
			tokens +=len(words)
			for w in words:
				if w not in vocab:
					vocab[w] = 1
				else:
					vocab[w] += 1
		idf_list = idf(art,vocab)
		#print idf_list
		voc = vocab.keys()
		n_vocab = len(vocab.keys())
		d = np.zeros([n_vocab,n_sent])

		s_index = 0
		for sent in art.allSentences:

			words = nltk.word_tokenize(sent.string)

			for w in words:
				w_index = voc.index(w)

				d[w_index][s_index] = count(words,w)
			s_index += 1

		a = np.dot(d,np.transpose(d))
		u,s,v = np.linalg.svd(a, full_matrices=False)
		K = int(math.ceil((0.1*(n_vocab))))
		s_prime = s[:K]
		med = np.median(s_prime)
		score.append(med)


	return score


def main():
		trainArticles = pd.preprocessData("developmentSet.dat", "developmentSetLabels.dat")
		svd_median = compute_tred_median(trainArticles)
		print svd_median




if __name__ == "__main__":
    main()