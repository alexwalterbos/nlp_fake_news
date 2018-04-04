import pickle as cp
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def generate_svd_feature(headlineTfidf, articleTfidf, train_num, test_num):
    svd = TruncatedSVD(n_components=50, n_iter=15)
    hBTfidf = vstack((headlineTfidf, articleTfidf))
    svd.fit(hBTfidf)
    headlineSvd = svd.transform(headlineTfidf)
    articleSvd = svd.transform(articleTfidf)

    simSvd = []
    for i in range(0, headlineTfidf.shape[0]):
        simSvd.append([cosine_similarity(headlineSvd[i].reshape(1, -1), articleSvd[i].reshape(1, -1))[0][0]])

    headlineSvdTrain = headlineSvd[:train_num, :]
    outfilename_train = "feature_pickles/svd.headline.train.pkl"
    with open(outfilename_train, "wb") as outfile:
        cp.dump(headlineSvdTrain, outfile, -1)

    if test_num > 0:
        headlineSvdTest = headlineSvd[train_num:, :]
        outfilename_test = "feature_pickles/svd.headline.test.pkl"
        with open(outfilename_test, "wb") as outfile:
            cp.dump(headlineSvdTest, outfile, -1)

    return simSvd
