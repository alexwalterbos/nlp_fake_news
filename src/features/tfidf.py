from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from util import test_set_size, train_set_size
import pickle


def generate_tfidf_feature(data):
    headline = data['Headline_unigram']
    article = data['articleBody_unigram']

    testNum = test_set_size(data)
    trainNum = train_set_size(data)

    # Number of train and test articles
    totalNum = len(headline)
    trainNum = totalNum - testNum

    # join the individual words into a sentence ({{'I', 'ate', 'pie'},...} becomes {{'I ate pie'},...})
    headlineNoVec = headline.map(lambda x: ' '.join(x)).tolist()
    articleNoVec = article.map(lambda x: ' '.join(x)).tolist()

    # Combine headlines and articles
    text_per_article = []
    for i in range(0, trainNum):
        text = headlineNoVec[i] + ' ' + articleNoVec[i]
        text_per_article.append(text)

    # First use a tfidfVectorizer to find all words
    vec = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2)
    vec.fit(text_per_article)
    vocabulary = vec.vocabulary_

    # Make different vectorizers for headlines and articles, it is necesary to use 2 different vectorizers
    vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2, vocabulary=vocabulary)
    vecA = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2, vocabulary=vocabulary)

    # Find Tfidf values for headlines and articles
    headlineTfidf = vecH.fit_transform(headlineNoVec)
    articleTfidf = vecA.fit_transform(articleNoVec)

    # Make seperate variables for taining and testing
    headlineTrain = headlineTfidf[:trainNum, :]
    articleTrain = articleTfidf[:trainNum, :]
    if testNum > 0:
        headlineTest = headlineTfidf[trainNum:, :]
        articleTest = articleTfidf[trainNum:, :]

    print(headlineTfidf.shape)

    # Find cosine similarities
    simVec = []
    for i in range(0, totalNum):
        simVec.append([cosine_similarity(headlineTfidf[i], articleTfidf[i])[0][0]])
    print(len(simVec))

    # make seperate variables for tain and test similarites.
    simTfidfTrain = simVec[:trainNum]
    simTfidfTest = simVec[trainNum:]

    return [headlineTfidf, articleTfidf, simVec]


def read(header='train'):
    filename_hvec = "feature_pickles/tfidf_head_%s" % header
    with open(filename_hvec, "rb") as infile:
        headlineVec = pickle.load(infile)

    filename_bvec = "feature_pickles/%tfidf_article_%s" % header
    with open(filename_bvec, "rb") as infile:
        bodyVec = pickle.load(infile)

    filename_simvec = "feature_pickles/tfidf_sim_%s" % header
    with open(filename_simvec, "rb") as infile:
        simVec = pickle.load(infile)

    return [headlineVec, bodyVec, simVec]
