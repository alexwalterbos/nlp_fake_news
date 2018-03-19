import gensim
import numpy as np
from sklearn.preprocessing import normalize
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def generate_tfidf_feature(data, numTest=0):
    #load model from google
    model = gensim.models.KeyedVectors.load_word2vec_format('data_files/GoogleNews-vectors-negative300.bin', binary=True)

    #load data
    with open('data_files/smalldata.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    headline = data['Headline_unigram']
    article = data['articleBody_unigram']

    #Number of train and test articles
    totalNum = len(headline)
    testNum = numTest
    trainNum = totalNum-testNum



    #For every every headline/article: look up the vectors for every word in it, add them up, and normalize those vectors.
    i = 0
    headlineVec=[]
    for x in headline:
        z=0
        for y in x:
            if y in model:
               z = z +np.add(model.get_vector(y), [0.]*300)
       headlineVec.append(z)
    headlineVec = list(headlineVec)
    headlineVec = normalize(headlineVec)

    i = 0
    articleVec=[]
    for x in article:
        z=0
       for y in x:
           if y in model:
               z = z +np.add(model.get_vector(y), [0.]*300)
       articleVec.append(z)
    articleVec = list(articleVec)
    articleVec = normalize(articleVec)

    #make seperate variables for tain and test headlines/articles, currently this is not used.
    headlineTrain = headlineVec[:trainNum, :]
    articleTrain = articleVec[:trainNum, :]
    if testNum > 0:
        headlineTest = headlineVec[trainNum:, :]
        articleTest = articleVec[trainNum:, :]

    #Calculate cosine-similarity between headlines and respective articles and make variables for train and test similarities.
    simVecTemp = cosine_similarity(headlineVec, articleVec)
    simVec = []
    for i in range(0, totalNum):
        simVec.append(simVecTemp[i, i])
    simVecTrain = simVec[:trainNum]
    if testNum > 0:
        simVecTest = simVec[trainNum:]

    #Store in pickles
    with open('feature_pickles/word2vec_head_train.pkl', "wb") as outfile:
        pickle.dump(headlineTrain, outfile, -1)
    with open('feature_pickles/word2vec_head_test.pkl', "wb") as outfile:
        pickle.dump(headlineTest, outfile, -1)
    with open('feature_pickles/word2vec_article_train.pkl', "wb") as outfile:
        pickle.dump(articleTrain, outfile, -1)
    with open('feature_pickles/word2vec_article_test.pkl', "wb") as outfile:
        pickle.dump(articleTest, outfile, -1)
    with open('feature_pickles/word2vec_sim_train.pkl', "wb") as outfile:
        pickle.dump(simVecTrain, outfile, -1)
    with open('feature_pickles/word2vec_sim_test.pkl', "wb") as outfile:
        pickle.dump(simVecTest, outfile, -1)
    print('made 6 pickle files: word2vec_head_train, word2vec_head_test, word2vec_article_train, word2vec_article_test, word2vec_sim_train, word2vec_sim_test')

def read(self, header='train'):
    filename_hvec = "word2vec_head_%s" % header
    with open(filename_hvec, "rb") as infile:
        headlineVec = pickle.load(infile)

    filename_bvec = "word2vec_article_%s" % header
    with open(filename_bvec, "rb") as infile:
        bodyVec = pickle.load(infile)

    filename_simvec = "word2vec_sim_%s" % header
    with open(filename_simvec, "rb") as infile:
        simVec = pickle.load(infile)

    return [headlineVec, bodyVec, simVec]