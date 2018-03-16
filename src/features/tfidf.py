from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np


with open('data_files/smalldata.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
headline = data['Headline_unigram']
article = data['articleBody_unigram'] 

trainNum = 100
testNum = 100
totalNum = trainNum+testNum

headlineNoVec = headline.map(lambda x: ' '.join(x))
articleNoVec = article.map(lambda x: ' '.join(x))

text_per_article = []
for i in range(0, totalNum):
    text = headlineNoVec[i] + ' ' + articleNoVec[i]
    text_per_article.append(text)


vec = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2)
vec.fit(text_per_article) # Tf-idf calculated on the combined training + test set
vocabulary = vec.vocabulary_

vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2, vocabulary=vocabulary)
vecA = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2, vocabulary=vocabulary)

headlineTfidf = vecH.fit_transform(headlineNoVec)
articleTfidf = vecA.fit_transform(articleNoVec)

headlineTrain = headlineTfidf[:trainNum, :]
articleTrain = articleTfidf[:trainNum, :]
if testNum > 0:
    headlineTest = headlineTfidf[trainNum:, :]
    articleTest = articleTfidf[trainNum:, :]

print(headlineTfidf.shape)


simVec = []
for i in range (0, totalNum):
    simVec.append([cosine_similarity(headlineTfidf[i], articleTfidf[i])[0][0]])
print(len(simVec))

simTfidfTrain = simVec[:trainNum]
simTfidfTest = simVec[trainNum:]

with open('feature_pickles/tfidf_head_train.pkl', "wb") as outfile:
    pickle.dump(headlineTrain, outfile, -1)
with open('feature_pickles/tfidf_head_test.pkl', "wb") as outfile:
    pickle.dump(headlineTest, outfile, -1)
with open('feature_pickles/tfidf_article_train.pkl', "wb") as outfile:
    pickle.dump(articleTrain, outfile, -1)
with open('feature_pickles/tfidf_article_test.pkl', "wb") as outfile:
    pickle.dump(articleTest, outfile, -1)
with open('feature_pickles/tfidf_sim_train.pkl', "wb") as outfile:
    pickle.dump(simTfidfTrain, outfile, -1)
with open('feature_pickles/tfidf_sim_test.pkl', "wb") as outfile:
    pickle.dump(simTfidfTest, outfile, -1)
print ('made 6 pickle files: head_train, head_test, article_train, article_test, sim_train, sim_test')