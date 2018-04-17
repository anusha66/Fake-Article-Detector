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

def SyntacticalFeatureExtractorFunction(Articles):

    regex_noun = re.compile('nn*')
    regex_verb = re.compile('vb*')

    regex1 = re.compile('jj*')
    regex2 = re.compile('nn*')
    regex3 = re.compile('vb*')
    regex4 = re.compile('rb*')

    regex5 = re.compile('prp*')
    regex6 = re.compile('in*')
    regex7 = re.compile('cc*')
    regex8 = re.compile('dt*')

    for i in range(len(Articles)):

        for k in range(len(Articles[i].allSentencesString)):

            text = Articles[i].allSentencesString[k].lower().split()

            Articles[i].allSentences[k].token = text

            if (len(re.findall(r'jj',Articles[i].allSentencesString[k].lower())) != 0):
                Articles[i].allSentences[k].adjNoun = float((len((re.findall(r'jj[r|s]? nn[s|p|ps]?',Articles[i].allSentencesString[k].lower()))))/len(re.findall(r'jj[r|s]?',Articles[i].allSentencesString[k].lower())))
            else:
                Articles[i].allSentences[k].adjNoun = 0

            text_n = [x for x in text if regex_noun.match(x)]
            Articles[i].allSentences[k].noun = len(text_n)

            text_v = [x for x in text if regex_verb.match(x)]
            Articles[i].allSentences[k].verb = len(text_v)

            if (float(len([x for x in text if regex8.match(x)]) + len([x for x in text if regex7.match(x)]) + len(
                    [x for x in text if regex6.match(x)]) + len([x for x in text if regex5.match(x)]))) != 0.0:
                    Articles[i].allSentences[k].contentToNonPOS = float(
                    len([x for x in text if regex1.match(x)]) + len([x for x in text if regex2.match(x)]) + len(
                        [x for x in text if regex3.match(x)]) + len([x for x in text if regex4.match(x)])) / float(
                    len([x for x in text if regex8.match(x)]) + len([x for x in text if regex7.match(x)]) + len(
                        [x for x in text if regex6.match(x)]) + len([x for x in text if regex5.match(x)]))
            else:
                    Articles[i].allSentences[k].contentToNonPOS = 152.0

    nounsRatio = []
    verbsRatio = []
    contentToNonPOSRatio = []
    adjNounRatio = []
    avgsentlenF = []

    for i in range(len(Articles)):

        #print(Articles[i].allSentencesString)

        tokens = []
        noun = 0
        verb = 0
        adjNoun = 0
        contentToNonPOS = 0

        for k in range(len(Articles[i].allSentencesString)):
            tokens = tokens + (Articles[i].allSentences[k].token)
            contentToNonPOS = contentToNonPOS + (Articles[i].allSentences[k].contentToNonPOS) * len(Articles[i].allSentences[k].token)
            noun = noun + (Articles[i].allSentences[k].noun *  len(Articles[i].allSentences[k].token))
            verb = verb + (Articles[i].allSentences[k].verb *  len(Articles[i].allSentences[k].token))
            adjNoun = adjNoun + Articles[i].allSentences[k].adjNoun * len(Articles[i].allSentences[k].token)

        types = list(set(tokens))

        Articles[i].token = tokens
        Articles[i].types = types

        Articles[i].avgsentlen = (len(tokens)) / Articles[i].numberOfSentences

        if len(tokens) != 0:
            Articles[i].adjNoun = float(adjNoun) / (len(tokens))
        else:
            Articles[i].adjNoun = 152.0

        if len(tokens) != 0:
            Articles[i].noun = (float(noun)) / (len(tokens))
        else:
            Articles[i].noun = 152.0

        if len(tokens) != 0:
            Articles[i].verb = (float(verb)) / (len(tokens))
        else:
            Articles[i].verb = 152.0

        if len(tokens) != 0:
            Articles[i].contentToNonPOS = (float(contentToNonPOS)) / (len(tokens))
        else:
            Articles[i].contentToNonPOS = 152.0

        nounsRatio.append(Articles[i].noun)
        verbsRatio.append(Articles[i].verb)
        contentToNonPOSRatio.append(Articles[i].contentToNonPOS)
        adjNounRatio.append(Articles[i].adjNoun)
        avgsentlenF.append(Articles[i].avgsentlen)

    return avgsentlenF, nounsRatio, verbsRatio, contentToNonPOSRatio, adjNounRatio