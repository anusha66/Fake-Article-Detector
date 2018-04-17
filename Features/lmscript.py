import os
import preprocessData as pd
import numpy as np



def main():
    trainArticles = pd.preprocessData("trainingSet.dat", "trainingSetLabels.dat")
    four_grams = []
    perplexity = []
    for article in trainArticles:
        fp = open('./temp_trained/red.txt','w')
        for sentence in article.allSentences:
            fp.write(sentence.string)
            fp.write('\n')
        fp.close()
        os.system('cd temp_trained/')
        os.system('echo "perplexity -text red.txt" | ./evallm -binary a.binlm')
        break

    print 'bla'




if __name__ == '__main__':
    main()