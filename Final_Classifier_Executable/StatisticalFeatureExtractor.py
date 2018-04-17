import preprocessData
# import language_check
from nltk.corpus import stopwords
import string
from nltk.stem.porter import *
from stop_words import get_stop_words

# Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()

# Punctuation Removal
exclude = set(string.punctuation)

# Stop Word List
stop = set(stopwords.words('english'))
stop2 = get_stop_words('en')
stop = list(stop)
stop.extend(list(stop2))
stop = set(stop)
stop = list(stop)

def get_stop_words(doc):

    stop_words = " ".join([i for i in doc.lower().split() if i in stop])
    return stop_words

def clean_document(doc):

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    return stop_free

def StatisticalFeatureExtractorFunction(Articles):

    for i in range(len(Articles)):

        for k in range(len(Articles[i].allSentencesString)):

            text = get_stop_words(Articles[i].allSentencesString[k])
            text = text.split()

            Articles[i].allSentences[k].stopwords = (len(text))

            text = clean_document(Articles[i].allSentencesString[k])
            text = text.split()

            Articles[i].allSentences[k].token = text
            Articles[i].allSentences[k].types = list(set(text))

            if (len(Articles[i].allSentences[k].token) != 0):
                Articles[i].allSentences[k].ttr = \
                    float(len(Articles[i].allSentences[k].types)) / len(Articles[i].allSentences[k].token)
            else:
                Articles[i].ttr = 152.0

    ttrRatio = []
    stopwordsRatio = []

    for i in range(len(Articles)):

        tokens = []
        count = 0

        for k in range(len(Articles[i].allSentencesString)):
            tokens = tokens + (Articles[i].allSentences[k].token)
            count = count + (Articles[i].allSentences[k].stopwords)

        types = list(set(tokens))

        Articles[i].token = tokens
        Articles[i].types = types

        if len(tokens) != 0:
            Articles[i].avgstopwords = float(count) / (len(tokens))
        else:
            Articles[i].avgstopwords = 152.0


        if len(tokens) != 0:
            Articles[i].ttr = float(len(types)) / len(tokens)
        else:
            Articles[i].ttr = 152.0

        ttrRatio.append(Articles[i].ttr)
        stopwordsRatio.append(Articles[i].avgstopwords)

    return ttrRatio, stopwordsRatio