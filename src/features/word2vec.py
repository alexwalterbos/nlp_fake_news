import gensim
import numpy as np
from sklearn.preprocessing import normalize
import pickle
from sklearn.metrics.pairwise import cosine_similarity

#load model from google
model = gensim.models.KeyedVectors.load_word2vec_format('data_files/GoogleNews-vectors-negative300.bin', binary=True)

#load data
with open('data_files/smalldata.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
headline = data['Headline_unigram']
article = data['articleBody_unigram']

#Number of train and test articles
trainNum = 100
testNum = 100
totalNum = trainNum+testNum


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
with open('feature_pickles/word2vec_sim_train.pkl', "wb") as outfile:
    pickle.dump(simVecTrain, outfile, -1)
with open('feature_pickles/word2vec_sim_test.pkl', "wb") as outfile:
    pickle.dump(simVecTest, outfile, -1)
print('made 2 pickle files: word2vec_sim_train, word2vec_sim_test')