import preprocessData as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import  word_tokenize
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
import nltk


def calculateEntropy(article_tokens):
    typeDict = dict()
    for token in article_tokens:
        if typeDict.get(token, None) is None:
            typeDict[token] = 1
        else:
            typeDict[token] += 1

    l = sorted(typeDict.iteritems(), key = lambda (k, v):(v, k), reverse=True)

    freqDict = dict()
    for item in l:
        count = freqDict.get(item[1], 0)
        freqDict[item[1]] = count + 1

    sum = 0.0
    N = len(article_tokens)

    for key in freqDict.keys():
        p = freqDict[key] * int(key)/float(N)
        sum += p * np.log2(p)

    if sum != 0.0:
        entropy = -1 * sum
    else:
        entropy = 0.0

    ratio_1 = freqDict.get(1, 0)/float(len(typeDict))
    ratio_2 = freqDict.get(2, 0)/float(len(typeDict))

    return entropy, ratio_1, ratio_2

def get_lexical_entropy(train_data):

    ps = PorterStemmer()

    entropy = []
    ratio1 = []
    ratio2 = []
    for article in train_data:
        article_tokens = []
        for sentence in article.allSentences:
            word_tokens = word_tokenize(sentence.string.upper())

            word_tokens = [ps.stem(w).upper() for w in word_tokens]
            article_tokens.extend(word_tokens)

        e, r1, r2 = calculateEntropy(article_tokens)
        entropy.append(e)
        ratio1.append(r1)
        ratio2.append(r2)


    fp = open('EmpericalEntropy_balanced_train.txt', 'w')
    for item in entropy:
        fp.write(str(item))
        fp.write('\n')

    fp.close()

    fp = open('ratio1_balanced_train.txt', 'w')
    for item in ratio1:
        fp.write(str(item))
        fp.write('\n')

    fp.close()

    fp = open('ratio2_balanced_train.txt', 'w')
    for item in ratio2:
        fp.write(str(item))
        fp.write('\n')

    fp.close()

    return entropy, ratio1, ratio2

def distanceMeasures():
    stop = set(stopwords.words('english'))
    stop2 = get_stop_words('en')
    stop = list(stop)
    stop.extend(list(stop2))
    stop = set(stop)
    stop = list(stop)

    def clean_stop(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i in stop])
        return stop_free

    articles = pd.preprocessData('developmentSet.dat', 'developmentSetLabels.dat')
    dist = []
    for article in articles:
        score = 0
        denom = 0
        for sentence in article.allSentences:
            s = nltk.word_tokenize(sentence.string)
            d = dict()
            for i, w in enumerate(s):
                if w.lower() in stop:
                    if w.upper() in d.keys():
                        d[w.upper()].append(i)
                    else:
                        d[w.upper()] = [i]

            temp = 0
            for key in d.keys():
                if d[key] > 1:
                    for i in range(1, len(d[key])):
                        temp += d[key][i] - d[key][i-1]

            score += sentence.length * temp
            denom += sentence.length

        ar_score = score/float(denom)
        dist.append(ar_score)

    return dist

    fp = open('distance_test.txt', 'w')

    for item in dist:
        fp.write(str(item))
        fp.write('\n')

    fp.close()
    print 'bla'



if __name__ == '__main__':
    train_data = pd.preprocessData('balancedTrainingData.dat', 'balancedTrainingDataLabels.dat')
    get_lexical_entropy(train_data)