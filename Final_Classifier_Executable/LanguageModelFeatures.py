import kenlm
import preprocessData as pd


def get_lang_features(trainArticles):
    # trainArticles = pd.preprocessData("balancedTrainingData.dat", "balancedTrainingDataLabels.dat")
    perplexity_true = []
    perplexity_fake = []
    perplexity = []

    # tri_model = kenlm.Model('model3g_fresh.arpa')
    # quad_model = kenlm.Model('model4g_fresh.arpa')
    # five_model = kenlm.Model('model5g_fresh.arpa')

    tri_model = kenlm.LanguageModel('model3g_fresh.binary')
    quad_model = kenlm.LanguageModel('model4g_fresh.binary')
    five_model = kenlm.LanguageModel('model5g_fresh.binary')

    ratioTriQuad = []
    ratioTriFive = []

    for article in trainArticles:
        num_sentences = article.numberOfSentences
        tri_score = 0
        quad_score = 0
        five_score = 0

        for sentence in article.allSentences:
            tri_score += float(tri_model.perplexity(sentence.string))
            quad_score += float(quad_model.perplexity(sentence.string))
            five_score += float(five_model.perplexity(sentence.string))

        tri_score = float(tri_score)/num_sentences
        quad_score = float(quad_score)/num_sentences
        five_score = float(five_score)/num_sentences

        ratioTriQuad.append(float(quad_score)/tri_score)
        ratioTriFive.append(float(five_score)/tri_score)

    return ratioTriQuad, ratioTriFive


        # perplexity.append(score)
        # if article.label == 0:
        #     perplexity_fake.append(score)
        # else:
        #     perplexity_true.append(score)

    # fp = open('ratioTriQuad_train.txt', 'w')

    # for item in ratioTriQuad:
    #     fp.write(str(item))
    #     fp.write('\n')

    # fp.close()

    # fp = open('ratioTriFive_train.txt', 'w')

    # for item in ratioTriFive:
    #     fp.write(str(item))
    #     fp.write('\n')

    # fp.close()



    # fp = open('model3g_fresh.arpa')
    # fp.readline()
    # fp.readline()
    # line = fp.readlines()
    # print line

if __name__ == '__main__':
    main()
