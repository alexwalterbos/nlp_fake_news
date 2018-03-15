from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

headline = ['queen', 'land', 'stuf', 'queen', 'king', 'sauce', 'a']
article = ['king', 'king', 'queen', 'land', 'stuf', 'stuf2']

trainNum = 5
testNum = 0

vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
vec.fit(headline+article) # Tf-idf calculated on the combined training + test set
vocabulary = vec.vocabulary_

vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)

headlineTfidf = vecH.fit_transform(headline)
articleTfidf = vecH.fit_transform(article)

headlineTrain = headlineTfidf[:trainNum, :]
articleTrain = articleTfidf[:trainNum, :]
if testNum > 0:
    headlineTest = headlineTfidf[trainNum:, :]
    articleTest = articleTfidf[trainNum:, :]
	
print(headlineTfidf)
print()
print(articleTfidf)

simTfidf = cosine_similarity(headlineTfidf, articleTfidf)
simTfidfTrain = simTfidf[:trainNum]
if testNum >0:
    simTfidfTest = simTfidf[trainNum:]
	
print(simTfidf)