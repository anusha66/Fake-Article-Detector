import kenlm
import preprocessData as pd
from bllipparser import RerankingParser
rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=False)

def main():
    global rrp
    trainArticles = pd.preprocessData("balancedTrainingData.dat", "balancedTrainingDataLabels.dat")
    perplexity_true = []
    perplexity_fake = []
    perplexity = []

    tri_model = kenlm.Model('fake_pos_2g.binary')
    quad_model = kenlm.Model('fake_pos_3g.binary')
    five_model = kenlm.Model('fake_pos_4g.binary')

    ratioTriQuad = []
    ratioTriFive = []

    for article in trainArticles:
        num_sentences = article.numberOfSentences
        tri_score = 0
        quad_score = 0
        five_score = 0

        for sentence in article.allSentences:
            pos_tags = rrp.tag(sentence.string)
            words,tags = zip(*pos_tags)
            sentence.string = ' '.join(tags)
            tri_score += float(tri_model.perplexity(sentence.string))
            quad_score += float(quad_model.perplexity(sentence.string))
            five_score += float(five_model.perplexity(sentence.string))

        tri_score = float(tri_score)/num_sentences
        quad_score = float(quad_score)/num_sentences
        five_score = float(five_score)/num_sentences

        ratioTriQuad.append(float(quad_score)/tri_score)
        ratioTriFive.append(float(five_score)/tri_score)


        # perplexity.append(score)
        # if article.label == 0:
        #     perplexity_fake.append(score)
        # else:
        #     perplexity_true.append(score)

    fp = open('ratioBiTri_pos_train.txt', 'w')

    for item in ratioTriQuad:
        fp.write(str(item))
        fp.write('\n')

    fp.close()

    fp = open('ratioBiQuad_pos_train.txt', 'w')

    for item in ratioTriFive:
        fp.write(str(item))
        fp.write('\n')

    fp.close()


if __name__ == '__main__':
    main()
