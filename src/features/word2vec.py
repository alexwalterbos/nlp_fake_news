import gensim
import numpy as np
from sklearn.preprocessing import normalize

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

numTrain = 4
numTest = 0

headline = ['queen', 'land', 'stuf', 'queen', 'king', 'a']
article = ['king', 'king', 'queen', 'land', 'stuf2', 'stuf2']

y=[]
headlineVec=[]
for x in headline:
    if x in model:
        y.append(np.add(model.get_vector(x), [0.]*300)) #is nodig want anders komt er metadata in mn array?
headlineVec = np.array(y)
headlineVec = normalize(headlineVec)

y=[]
articleVec=[]
for x in article:
    if x in model:
        y.append(np.add(model.get_vector(x), [0.]*300)) #is nodig want anders komt er metadata in mn array?
articleVec = np.array(y)
articleVec = normalize(articleVec)

print (headlineVec.shape)
print (articleVec.shape)

headlineTrain = headlineVec[:numTrain, :]
articleTrain = articleVec[:numTrain, :]
if numTest > 0:
    headlineTest = headlineVec[numTrain:, :]
    articleTest = articleVec[numTrain:, :]

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity(headlineVec, articleVec))
simVec = cosine_similarity(headlineVec, articleVec)
simVecTrain = simVec[:numTrain]
if numTest > 0:
    simVecTest = simVec[numTrain:]

print(simVec)