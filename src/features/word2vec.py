import gensim
import numpy as np
from sklearn.preprocessing import normalize
import pickle
from sklearn.metrics.pairwise import cosine_similarity


model = gensim.models.KeyedVectors.load_word2vec_format('data_files/GoogleNews-vectors-negative300.bin', binary=True)

with open('data_files/smalldata.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
headline = data['Headline_unigram']
article = data['articleBody_unigram']

trainNum = 100
testNum = 100
totalNum = trainNum+testNum


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
articleVec = list(headlineVec)
articleVec = normalize(headlineVec)


headlineTrain = headlineVec[:trainNum, :]
articleTrain = articleVec[:trainNum, :]
if testNum > 0:
    headlineTest = headlineVec[trainNum:, :]
    articleTest = articleVec[trainNum:, :]


simVecTemp = cosine_similarity(headlineVec, articleVec)
simVec = []
for i in range(0, totalNum):
    simVec.append(simVecTemp[i, i])
simVecTrain = simVec[:trainNum]
if testNum > 0:
    simVecTest = simVec[trainNum:]
print(len(simVec))

with open('feature_pickles/word2vec_sim_train.pkl', "wb") as outfile:
    pickle.dump(simVecTrain, outfile, -1)
with open('feature_pickles/word2vec_sim_test.pkl', "wb") as outfile:
    pickle.dump(simVecTest, outfile, -1)