import preprocessData as pd
############ Parser Score #################
import math,nltk
import pickle
from bllipparser import RerankingParser
from multiprocessing import Pool
rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)

def sent_struct(articles):
    
    fake_score = []
    real_score = []
    gram_scores = []
    score_work = []
    count = 0

    agents = 20
    chunksize = 3
    
    p = Pool(agents)
    result = p.map(calculate_score, articles)
    print result

#    gram_scores.append(sc/float(lc))
#    score_work.append([article.label,sc/float(lc)])

    #    print "--------------------------------"
    pickle_file = open('article_gram_score.pkl','wb')
    pickle.dump(result,pickle_file)
    pickle_file.close()
    # pickle_file = open('score.pkl','wb')
    # pickle.dump(score_work,pickle_file)
    # pickle_file.close()



def calculate_score(article):
        sc = 0
        lc = 0
        rrp = RerankingParser.from_unified_model_dir('/Users/anushreekumar/.local/share/bllipparser')
    #    print "~~~~~~~ Article ~~~~~~~~~~~"
        for sent in article.allSentences:
        #    print sent.string
            try:
                best_list = rrp.parse(sent.string)
                score = best_list[0].parser_score
            except Exception as e:
                score = -1000
                print str(e)
            sc += score*sent.length
            lc += sent.length
        #    sent_score.append(sc)
        #    print sent.length,score/float(sent.length),article.label
        
    #    print "Grammaticality score of ",article.label," article : ",sc/float(lc)
        return sc/float(lc)

########## POS Tagging ##############

def pos_tagged(articles,filename):
	f = open(filename,'w')
	for art in articles:
		f.write('~~~~~'+'\n')
		for sent in art.allSentences:
			text = nltk.word_tokenize(sent.string)
		#	pos_tags = rrp.tag(text)
			pos_tags = nltk.pos_tag(text)
			words,tags = zip(*pos_tags)
			#print tags
			tags = ' '.join(tags)
			print tags+'\n'
			f.write(tags+'\n')



def main():

    #### POS Tagging ######
    #tag_train = "tagged_train.txt"
    #nltk_train = "nltk_train.txt"
    #tag_test = "tagged_dev.txt"
    #nltk_test = "nltk_test.txt"
    #lmtrain = "lmtrain.txt"
    #pos_tagged(trainArticles,nltk_train)
    #pos_tagged(testArticles,nltk_test)
    #pos_tagged(corpus,lmtrain)

    #### Parser Score ########

    trainArticles = pd.preprocessData("trainingSet.dat", "trainingSetLabels.dat")
    sent_struct(trainArticles)
    #testArticles = preprocessData("developmentSet.dat", "developmentSetLabels.dat")
    #sent_struct(testArticles)

if __name__ == "__main__":
    main()