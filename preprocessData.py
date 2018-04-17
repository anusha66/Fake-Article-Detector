#Sentence Object - Add more attributes as required
class sentenceObj(object):

    def __init__(self, string="", length=0):
        self.string = string
        self.length = length

#Article Object - Add more attributes as required
class articleObj(object):

    def __init__(self, numberOfSentences=0, label=-1, allSentences=[]):
        self.numberOfSentences = numberOfSentences
        self.label = label
        self.allSentences = allSentences

#Creats an Article
def createArticle(numberOfSentences, label, allSentences):

    article = articleObj(numberOfSentences, label, allSentences)
    return article

#Creats a Sentence
def createSentence(string, length):

    sent = sentenceObj(string, length)
    return sent

#Preprocess Data
def preprocessData(data, labels):

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
                allArticles.append(createArticle(num, Labels[iteration], allSentences))
                iteration = iteration + 1
            else:
                k = 1

            num = 0
            allSentences = []

        else:

            l = l.lstrip("<s>")
            l = l.lstrip()

            l = l.rstrip("</s>")
            l = l.rstrip()

            allSentences.append(createSentence(l,len(l)))

            num = num + 1

    allArticles.append(createArticle(num, Labels[iteration], allSentences))

    return allArticles

def main():

    print "Train"
    trainArticles = preprocessData("trainingSet.dat","trainingSetLabels.dat")
    print len(trainArticles), "Number of Articles"
    print "Dev"
    testArticles = preprocessData("developmentSet.dat", "developmentSetLabels.dat")
    print len(testArticles), "Number of Articles"

if __name__ == "__main__":
    main()