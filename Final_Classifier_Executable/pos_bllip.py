import preprocessData as pd
import nltk
from bllipparser import RerankingParser

rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
from multiprocessing import Pool


def pos_tagged(articles, sfilename, filename):
    agents = 20
    chunksize = 5

    p = Pool(agents)
    f = open(sfilename, 'r')
    g = open(filename, 'w')
    # rrp = RerankingParser.from_unified_model_dir('/afs/andrew.cmu.edu/usr9/apkumar/.local/share/bllipparser')
    #    for art in articles:
    #        f.write('~~~~~'+'\n')
    #	text = []
    #	for sent in art.allSentences:
    #		text.append(sent.string)
    print "Done getting list of all sentences...."
    data = f.readlines()
    text = []
    data = data[68923:72923]
    for line in data:
        line = line.strip('</s>')
        text.append(line)

    result = p.map(tokenize, text)
    for sent in result:
        g.write(sent + '\n')


def tokenize(sentence):
    rrp = RerankingParser.from_unified_model_dir('/Users/pranavipotharaju/.local/share/bllipparser')

    sentence = sentence.rstrip("</s>")

    # if len(sentence) >= 399:
    #     words = nltk.word_tokenize(sentence)
    #     pos_tags = nltk.pos_tag(words)
    # else:
    #     pos_tags = rrp.tag(sentence)
    try:
        pos_tags = rrp.tag(sentence)
    except Exception as e:
        print 'blaaa'
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        print str(e)
    words, tags = zip(*pos_tags)
    tags = ' '.join(tags)
    return tags + '\n'


def main():
    #### POS Tagging ######
    # tag_train = "tagged_train.txt"
    # nltk_train = "nltk_train.txt"
    # tag_test = "tagged_dev.txt"
    # nltk_test = "nltk_test.txt"
    # lmtrain = "lmtrain.txt"
    # pos_tagged(trainArticles,nltk_train)
    # pos_tagged(testArticles,nltk_test)
    # pos_tagged(corpus,lmtrain)

    #### Parser Score ########

    trainArticles = pd.preprocessData("balancedTrainingData.dat", "balancedTrainingDataLabels.dat")
    filename = "lmtrain_pos_bllip.txt"
    sfilename = "corpus.txt"
    #   pos_tagged(trainArticles,filename)
    #    testArticles = pd.preprocessData("developmentSet.dat", "developmentSetLabels.dat")
    #    sent_struct(testArticles,filename)
    pos_tagged(trainArticles, sfilename, filename)


if __name__ == "__main__":
    main()
