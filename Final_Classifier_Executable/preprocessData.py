import StatisticalFeatureExtractor
import SyntacticalFeatureExtractor

#Sentence Object - Add more attributes as required
class sentenceObj(object):

    def __init__(self, string="", length=0, types = [], token = [], avgstopwords = 0, ttr=0, verb=0,  noun = 0, contentToNonPOS = 0, adjNoun = 0):
        self.string = string
        self.length = length

        self.types = types
        self.token = token

        self.ttr = ttr
        self.avgstopwords = avgstopwords

        self.noun = noun
        self.verb = verb
        self.contentToNonPOS = contentToNonPOS
        self.adjNoun = adjNoun

#Article Object - Add more attributes as required
class articleObj(object):

    def __init__(self, numberOfSentences=0, label=-1, allSentences=[], allSentencesString = [], avgsentlen =0, types = [], token = [], ttr=0, stopwords = 0, verb =0, noun = 0, contentToNonPOS = 0, adjNoun = 0):
        self.numberOfSentences = numberOfSentences
        self.label = label
        self.allSentences = allSentences
        self.allSentencesString = allSentencesString

        self.types = types
        self.token = token

        self.ttr = ttr
        self.avgsentlen = avgsentlen
        self.stopwords = stopwords

        self.noun = noun
        self.verb = verb
        self.contentToNonPOS = contentToNonPOS
        self.adjNoun = adjNoun

#Creats an Article
def createArticle(numberOfSentences, label, allSentences, allSentencesString):

    article = articleObj(numberOfSentences, label, allSentences, allSentencesString)
    return article

#Creats a Sentence
def createSentence(string, length):

    sent = sentenceObj(string, length)
    return sent

#Preprocess Data
def preprocessDataFunction(data, labels):

    Labels = []
    with open(labels) as f:
        lines = f.readlines()

        for l in lines:
            l = l.strip()
            Labels.append(int(l))

    f.close()

    trainSet = []
    with open(data) as f:
        lines = f.readlines()

        for l in lines:
            l = l.strip()
            trainSet.append(l)

    f.close()

    iteration = 0
    global num
    global k
    k = 0
    allArticles = []

    for l in trainSet:

        if l == "~~~~~":

            if (k != 0):
                allArticles.append(createArticle(num, Labels[iteration], allSentences, allSentencesString))
                iteration = iteration + 1
            else:
                k = 1

            num = 0
            allSentences = []
            allSentencesString = []

        else:

            l = l.lstrip("<s>")
            l = l.lstrip()

            l = l.rstrip("</s>")
            l = l.rstrip()

            allSentences.append(createSentence(l,len(l.split())))
            allSentencesString.append(l)

            num = num + 1

    allArticles.append(createArticle(num, Labels[iteration], allSentences, allSentencesString))
    return allArticles

def main():

    print("Train")
    trainArticles = preprocessDataFunction("balancedTrainingData_pos.dat","balancedTrainingDataLabels.dat")
    print(len(trainArticles), "Number of Articles")
    print("Dev")
    testArticles = preprocessDataFunction("test_pos_bllip.txt", "developmentSetLabels.dat")
    print(len(testArticles), "Number of Articles")

    avgsentlenF, nounsRatio, verbsRatio, contentToNonPOSRatio, adjNounRatio = SyntacticalFeatureExtractor.SyntacticalFeatureExtractorFunction(trainArticles)
    avgsentlenF, nounsRatio, verbsRatio, contentToNonPOSRatio, adjNounRatio = SyntacticalFeatureExtractor.SyntacticalFeatureExtractorFunction(testArticles)

    print("Train")
    trainArticles = preprocessDataFunction("balancedTrainingData.dat","balancedTrainingDataLabels.dat")
    print(len(trainArticles), "Number of Articles")
    print("Dev")
    testArticles = preprocessDataFunction("developmentSet.dat", "developmentSetLabels.dat")
    print(len(testArticles), "Number of Articles")

    ttrRatio, stopwordsRatio = StatisticalFeatureExtractor.StatisticalFeatureExtractorFunction(trainArticles)
    ttrRatio, stopwordsRatio = StatisticalFeatureExtractor.StatisticalFeatureExtractorFunction(testArticles)

    return trainArticles, testArticles

if __name__ == "__main__":
    main()