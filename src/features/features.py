from .sentiment import generate_sentiment_analysis_files

def extract_features(data): # Data must be of the type described in https://github.com/alexwalterbos/nlp_fake_news/issues/9
    # TODO Get feature vectors (count, tfidf, svd, word2vec, sentiment) for data
    sentiment = generate_sentiment_analysis_files(data)
    # print('sentiment.shape: {}'.format(sentiment.shape))
    print(sentiment)

    print('done')
