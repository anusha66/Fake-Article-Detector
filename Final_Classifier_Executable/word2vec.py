from gensim.models import KeyedVectors
import numpy as np

def word2vec(train_data):


		print "Loading Word2Vec............."

		filename = 'GoogleNews-vectors-negative300.bin'
		model = KeyedVectors.load_word2vec_format(filename, binary=True)
		print "[INFO] Model Successfully Retrieved!"
		w2v = dict(zip(model.wv.index2word, model.wv.syn0))

		dim = len(w2v.itervalues().next())

		print "Obtaining features now...."
		x_true_word2vec = []
		x_fake_word2vec = []

		for article in train_data['True']:
			x_true_word2vec.append(np.array([[w2v[word.lower()] for word in train_data['True'][article][sentence_idx].split(" ") if word.lower() in w2v] or [np.zeros(dim)] for sentence_idx in range(len(train_data['True'][article]))]))


		for article in train_data['Fake']:
			x_fake_word2vec.append(np.array([[w2v[word.lower()] for word in train_data['Fake'][article][sentence_idx].split(" ") if word.lower() in w2v] or [np.zeros(dim)] for sentence_idx in range(len(train_data['Fake'][article]))]))

		x_true_word2vec, x_fake_word2vec = np.array(x_true_word2vec), np.array(x_fake_word2vec)

		print "Converting to mean embeddings...."

		X_true_mean_embeddings = []
		X_fake_mean_embeddings = []
		labels_w2v = []
		for article_true in x_true_word2vec:
			# sum_article = np.var([np.mean(sentence) for sentence in article_true])
			sum_article = np.sum([np.sum(sentence) for sentence in article_true])
			
			label_article = [1 for sentence in article_true]
			len_article = np.sum([len(sentence) for sentence in article_true])

			# X_true_mean_embeddings.append([sum_article])
			X_true_mean_embeddings.append(sum_article/len_article)
			# labels_w2v.extend(label_article)

		for article_fake in x_fake_word2vec:
			# sum_article = np.var([np.mean(sentence) for sentence in article_fake])
			
			sum_article = np.sum([np.sum(sentence) for sentence in article_fake])
			
			label_article = [0 for sentence in article_fake]
			len_article = np.sum([len(sentence) for sentence in article_fake])

			# X_fake_mean_embeddings.append([sum_article])
			X_fake_mean_embeddings.append(sum_article/len_article)
			# labels_w2v.extend(label_article)
		
		return X_true_mean_embeddings + X_fake_mean_embeddings
