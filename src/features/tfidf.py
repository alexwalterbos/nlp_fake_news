from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# load data
with open('data_files/smalldata.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
headline = data['Headline_unigram']
article = data['articleBody_unigram']

#Number of train and test articles
trainNum = 100
testNum = 100
totalNum = trainNum+testNum

#join the individual words into a sentence ({{'I', 'ate', 'pie'},...} becomes {{'I ate pie'},...})
headlineNoVec = headline.map(lambda x: ' '.join(x))
articleNoVec = article.map(lambda x: ' '.join(x))

#Combine headlines and articles
text_per_article = []
for i in range(0, totalNum):
    text = headlineNoVec[i] + ' ' + articleNoVec[i]
    text_per_article.append(text)

#First use a tfidfVectorizer to find all words
vec = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2)
vec.fit(text_per_article) # Tf-idf calculated on the combined training + test set
vocabulary = vec.vocabulary_

#Make different vectorizers for headlines and articles, it is necesary to use 2 different vectorizers
vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2, vocabulary=vocabulary)
vecA = TfidfVectorizer(ngram_range=(1, 3), max_df=.8, min_df=2, vocabulary=vocabulary)

#Find Tfidf values for headlines and articles
headlineTfidf = vecH.fit_transform(headlineNoVec)
articleTfidf = vecA.fit_transform(articleNoVec)

#Make seperate variables for taining and testing
headlineTrain = headlineTfidf[:trainNum, :]
articleTrain = articleTfidf[:trainNum, :]
if testNum > 0:
    headlineTest = headlineTfidf[trainNum:, :]
    articleTest = articleTfidf[trainNum:, :]

print(headlineTfidf.shape)

#Find cosine similarities
simVec = []
for i in range (0, totalNum):
    simVec.append([cosine_similarity(headlineTfidf[i], articleTfidf[i])[0][0]])
print(len(simVec))

#make seperate variables for tain and test similarites.
simTfidfTrain = simVec[:trainNum]
simTfidfTest = simVec[trainNum:]

#store in pickle files
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