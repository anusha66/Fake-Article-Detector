import preprocessData as pd
import numpy as np


def type_token(trainArticles):
    true = dict()
    fake = dict()
    count = 0
    for article in trainArticles:
        tokenCount = 0
        types = set()
        for sentence in article.allSentences:
            tokens = sentence.string.rstrip().split(' ')
            types = types.union(set(tokens))
            tokenCount += len(tokens)

        ratio = float(len(types))/tokenCount

        if article.label == 0:
            fake[count] = ratio
        else:
            true[count] = ratio

        count += 1

    tt = np.array(true)
    sd_t = np.std(tt)
    sd_f = np.std(np.array(fake))
    print 'bla'


def type_token_whole(trainArticles):
    trueTokens = []
    fakeTokens = []
    for article in trainArticles:
        tokens = []
        for sentence in article.allSentences:
            tokens_inner = sentence.string.rstrip().split(' ')
            tokens.extend(tokens_inner)

        if article.label == 0:
            trueTokens.extend(tokens)
        else:
            fakeTokens.extend(tokens)

    trueTypes = set(trueTokens)
    fakeTypes = set(fakeTokens)

    trueDict = dict()
    for token in trueTokens:
        if trueDict.get(token, 0) == 0:
            trueDict[token] = 1
        else:
            trueDict[token] += 1

    fakeDict = dict()
    for token in fakeTokens:
        if fakeDict.get(token, 0) == 0:
            fakeDict[token] = 1
        else:
            fakeDict[token] += 1


    print 'true ratio:', float(len(trueTokens))/len(trueTypes)
    print 'fake ratio:', float(len(fakeTokens)) / len(fakeTypes)


def main():
    trainArticles = pd.preprocessData("trainingSet.dat", "trainingSetLabels.dat")
    type_token(trainArticles)
    type_token_whole(trainArticles)

if __name__ == '__main__':
    main()