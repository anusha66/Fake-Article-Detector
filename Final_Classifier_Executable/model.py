#!/usr/bin/env python

from Load_Data import Load_Data, Load_Test_Data
import preprocessData as pd, numpy as np, sys
from StatisticalFeatureExtractor import StatisticalFeatureExtractorFunction
from SyntacticalFeatureExtractor import SyntacticalFeatureExtractorFunction
from MedianSVD_feat import compute_tred_median
from topical_redundancy import compute_tred
from sklearn.externals import joblib
from word2vec import word2vec
from lexicalEntropy import get_lexical_entropy
from LanguageModelFeatures import get_lang_features
import csv

if __name__ == "__main__":

	test_file = sys.argv[1]
	test_output_file = sys.argv[2]

	Load_Test_Data(test_file)

	X = []

	testData = pd.preprocessDataFunction(test_file, "dummy-labels.txt")

	ttr, stopWordsRatio = StatisticalFeatureExtractorFunction(testData)
	print "Stat features done!"

	entropy, ratio1, ratio2 = get_lexical_entropy(testData)

	avgsentlenF, nounsRatio, verbsRatio, contentToNonPOSRatio, adjNounRatio = SyntacticalFeatureExtractorFunction(testData)
	print "Syn features done!"
	
	ratioTriQuad, ratioTriFive = get_lang_features(testData)
	print "Lang Features Done!"

	SVDMedian = compute_tred_median(testData)
	print "Median features done!"
	
	loss,stats,sparse = compute_tred(testData)
	
	print "top_red features done!"
	
	# testData_word2vec = Load_Data(test_file,"dummy-labels.txt")

	# mean_word2vec = word2vec(testData_word2vec.return_train_data())
	# print "word2vec done!"

	X.append(avgsentlenF)
	X.append(ttr)
	X.append(stopWordsRatio)
	X.append(ratioTriFive)
	X.append(ratioTriQuad)
	X.append(entropy)
	X.append(ratio1)
	X.append(ratio2)
	X.append(SVDMedian)

	X = np.array(X)

	X = X.T[:,:]
	model = joblib.load('lr_best.pkl')
	y_predicted = model.predict_proba(X)

	with open(test_output_file,"w") as output_file:
		for prediction in y_predicted:
			if prediction[0] > prediction[1]:
				label = 1
			else:
			 	label = 0
			print prediction[0], prediction[1], label
			output_file.write(str(prediction[0])+" "+str(prediction[1])+" "+str(label)+"\n")




