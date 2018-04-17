import kenlm
import preprocessData as pd


def main():
    trainArticles = pd.preprocessData("nltk_train.txt", "trainingSetLabels.dat")
    perplexity_true = []
    perplexity_fake = []
    perplexity = []
    model = kenlm.Model('pos4g_fresh.arpa')

    for article in trainArticles:
        num_sentences = article.numberOfSentences
        score = 0
        for sentence in article.allSentences:
            score += float(model.perplexity(sentence.string))

        score = float(score)/num_sentences
        perplexity.append(score)
        if article.label == 0:
            perplexity_fake.append(score)
        else:
            perplexity_true.append(score)

    fp = open('pos_4_train.txt', 'w')

    for item in perplexity:
        fp.write(str(item))
        fp.write('\n')

    fp.close()


if __name__ == '__main__':
    main()
