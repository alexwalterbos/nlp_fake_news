# note that you need to download the word2vec bin from google before you run this class, for more information see data_files\readme.txt

import pickle

import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def generate_word2vec_feature(data, numTest=0):
    # load model from google
    model = gensim.models.KeyedVectors.load_word2vec_format('data_files/GoogleNews-vectors-negative300.bin', binary=True, limit=1000)

    headline = data['Headline_unigram']
    article = data['articleBody_unigram']

    # Number of train and test articles
    totalNum = len(headline)
    testNum = numTest
    trainNum = totalNum - testNum

    # For every every headline/article: look up the vectors
    # for every word in it, add them up, and normalize those vectors.
    i = 0
    headlineVec = []
    for x in headline:
        z = [0.] *300
        for y in x:
            if y in model:
                z = z + np.add(model.get_vector(y), [0.] * 300)
        headlineVec.append(z)

    headlineVec = list(headlineVec)
    headlineVec = normalize(headlineVec)

    i = 0
    articleVec = []
    for x in article:
        z = [0.] * 300
        for y in x:
            if y in model:
                z = z + np.add(model.get_vector(y), [0.] * 300)
        articleVec.append(z)
    articleVec = list(articleVec)
    articleVec = normalize(articleVec)

    # make seperate variables for tain and test headlines/articles, currently this is not used.
    headlineTrain = headlineVec[:trainNum, :]
    articleTrain = articleVec[:trainNum, :]
    if testNum > 0:
        headlineTest = headlineVec[trainNum:, :]
        articleTest = articleVec[trainNum:, :]

    # Calculate cosine-similarity between headlines and respective articles and make variables for train and test similarities.
    simVec = []
    for i in range(0, len(headlineVec)):
        sim = cosine_similarity([headlineVec[i]], [articleVec[i]])
        for s in sim:
            for s2 in s:
                simVec.append(s2)
    simVec = np.asarray(simVec)[:, np.newaxis]
    simVecTrain = simVec[:trainNum]
    if testNum > 0:
        simVecTest = simVec[trainNum:]

    return [headlineVec, articleVec, simVec]


def read(header='train'):
    """filename_hvec = "feature_pickles/word2vec_head_%s" % header
    with open(filename_hvec, "rb") as infile:
        headlineVec = pickle.load(infile)

    filename_bvec = "feature_pickles/word2vec_article_%s" % header
    with open(filename_bvec, "rb") as infile:
        bodyVec = pickle.load(infile)
    """

    filename_simvec = "feature_pickles/word2vec_sim_%s.pkl" % header
    with open(filename_simvec, "rb") as infile:
        simVec = pickle.load(infile)

    #return [headlineVec, bodyVec, simVec]
    return simVec
