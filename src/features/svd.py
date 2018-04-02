import pickle as cp
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

trainNum = 100
testNum = 100
totalNum = trainNum + testNum


def generate_svd_feature(h_tfidf_train, a_tfidf_train, h_tfidf_test, a_tfidf_test):
    # with open('feature_pickles/tfidf_head_train.pkl', 'rb') as file:
    #     headlineTfidfTrain = cp.load(file, encoding="latin1")
    #
    # with open('feature_pickles/tfidf_article_train.pkl', 'rb') as file:
    #     articleTfidfTrain = cp.load(file, encoding="latin1")

    headlineTfidf = h_tfidf_train
    articleTfidf = a_tfidf_train

    if testNum > 0:
        # with open('feature_pickles/tfidf_head_test.pkl', 'rb') as file:
        #     headlineTfidfTest = cp.load(file, encoding="latin1")
        #
        # with open('feature_pickles/tfidf_article_test.pkl', 'rb') as file:
        #     articleTfidfTest = cp.load(file, encoding="latin1")

        headlineTfidf = vstack((h_tfidf_train, h_tfidf_test))
        articleTfidf = vstack((a_tfidf_train, a_tfidf_test))

    svd = TruncatedSVD(n_components=50, n_iter=15)
    hBTfidf = vstack((headlineTfidf, articleTfidf))
    svd.fit(hBTfidf)
    headlineSvd = svd.transform(headlineTfidf)
    articleSvd = svd.transform(articleTfidf)

    simSvd = []
    for i in range(0, totalNum):
        simSvd.append([cosine_similarity(headlineSvd[i].reshape(1, -1), articleSvd[i].reshape(1, -1))[0][0]])

    headlineSvdTrain = headlineSvd[:trainNum, :]
    outfilename_train = "feature_pickles/svd.headline.train.pkl"
    with open(outfilename_train, "wb") as outfile:
        cp.dump(headlineSvdTrain, outfile, -1)

    if testNum > 0:
        headlineSvdTest = headlineSvd[trainNum:, :]
        outfilename_test = "feature_pickles/svd.headline.test.pkl"
        with open(outfilename_test, "wb") as outfile:
            cp.dump(headlineSvdTest, outfile, -1)

    articleSvdTrain = articleSvd[:trainNum, :]
    outfilename_train = "feature_pickles/svd.article.train.pkl"
    with open(outfilename_train, "wb") as outfile:
        cp.dump(articleSvdTrain, outfile, -1)

    if testNum > 0:
        articleSvdTest = articleSvd[trainNum:, :]
        outfilename_test = "feature_pickles/svd.article.test.pkl"
        with open(outfilename_test, "wb") as outfile:
            cp.dump(articleSvdTest, outfile, -1)

    simSvdTrain = simSvd[:trainNum]
    outfilename_train = "feature_pickles/svd.sim.train.pkl"
    with open(outfilename_train, "wb") as outfile:
        cp.dump(simSvdTrain, outfile, -1)

    if testNum > 0:
        simSvdTest = simSvd[trainNum:]
        outfilename_test = "feature_pickles/svd.sim.test.pkl"
        with open(outfilename_test, "wb") as outfile:
            cp.dump(simSvdTest, outfile, -1)
