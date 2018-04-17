

class Load_Test_Data():
	def __init__(self, train_data_filename):
		self.train_data = {'True':{},'Fake':{}} 
		self.sentences_list = []
		self.true_sentences = []
		self.fake_sentences = []
		self.labels = []
		train_label_filename = "dummy-labels.txt"
		with open(train_data_filename,"r") as f_data, open(train_label_filename,"w") as f_labels:
			# true_article_count, fake_article_count,label = 0,0,-1
			for train_point in f_data:	
				if train_point.strip() == '~~~~~':
					f_labels.write("1\n")
					# label = int(f_labels.readline())
					# if label:
					# 	self.labels.append(1)
					# 	true_article_count += 1
					# 	self.train_data['True'][true_article_count] = []
					# else:
					# 	self.labels.append(0)
					# 	fake_article_count += 1
					# 	self.train_data['Fake'][fake_article_count] = []
				# else:
				# 	self.sentences_list.append(" ".join(train_point.strip().split(" ")[1:-1]))
				# 	if label:
				# 		self.true_sentences.append(" ".join(train_point.strip().split(" ")[1:-1]))
				# 		self.train_data['True'][true_article_count].append(" ".join(train_point.strip().split(" ")[1:-1]))
				# 	else:
				# 		self.fake_sentences.append(" ".join(train_point.strip().split(" ")[1:-1]))
				# 		self.train_data['Fake'][fake_article_count].append(" ".join(train_point.strip().split(" ")[1:-1]))
	
	# def return_train_data(self):
	# 	return self.train_data

	# def return_sentences(self):
	# 	return self.sentences_list

	# def return_true_fake(self):
	# 	return self.true_sentences, self.fake_sentences, self.labels

class Load_Data():
	def __init__(self, train_data_filename, train_label_filename):
		self.train_data = {'True':{},'Fake':{}} 
		self.sentences_list = []
		self.true_sentences = []
		self.fake_sentences = []
		self.labels = []
		with open(train_data_filename,"r") as f_data, open(train_label_filename,"r") as f_labels:
			true_article_count, fake_article_count,label = 0,0,-1
			for train_point in f_data:	
				if train_point.strip() == '~~~~~':
					label = int(f_labels.readline())
					if label:
						self.labels.append(1)
						true_article_count += 1
						self.train_data['True'][true_article_count] = []
					else:
						self.labels.append(0)
						fake_article_count += 1
						self.train_data['Fake'][fake_article_count] = []
				else:
					self.sentences_list.append(" ".join(train_point.strip().split(" ")[1:-1]))
					if label:
						self.true_sentences.append(" ".join(train_point.strip().split(" ")[1:-1]))
						self.train_data['True'][true_article_count].append(" ".join(train_point.strip().split(" ")[1:-1]))
					else:
						self.fake_sentences.append(" ".join(train_point.strip().split(" ")[1:-1]))
						self.train_data['Fake'][fake_article_count].append(" ".join(train_point.strip().split(" ")[1:-1]))
	
	def return_train_data(self):
		return self.train_data

	def return_sentences(self):
		return self.sentences_list

	def return_true_fake(self):
		return self.true_sentences, self.fake_sentences, self.labels