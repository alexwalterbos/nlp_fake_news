from features.sentiment import generate_sentiment_analysis_files
from features.count import generate_count_feature
from features.tfidf import generate_tfidf_feature
from features.word2vec import generate_word2vec_feature
from features.svd import generate_svd_feature
import numpy as np
from util import test_set_size, train_set_size


def extract_features(data):
    print('Generating count feature')
    count = generate_count_feature(data)

    print('Generating tfidf feature')
    [headlineTfidf, bodyTfidf, tfidfVec] = generate_tfidf_feature(data)

    print('Generating word2vec feature')
    [headlineVec, articleVec, wordVec] = generate_word2vec_feature(data, train_set_size(data))


    # print('Generating sentiment feature')
    [headlineSent, articleSent, sentVec] = generate_sentiment_analysis_files(data)

    test_size = test_set_size(data)
    train_size = train_set_size(data)

    print('Generating svd feature')
    svd = generate_svd_feature(
        headlineTfidf,
        bodyTfidf,
        test_size,
        train_size
    )
    #
    # with open('feature_pickles/train.pkl', "wb") as outfile:
    #     pickle.dump(wordVec[:train_set_size(data)], outfile, -1)
    #
    # with open('feature_pickles/test.pkl', "wb") as outfile:
    #     pickle.dump(wordVec[train_set_size(data):], outfile, -1)

    print('done generating features')

    features = (
        count,
        # headlineTfidf, bodyTfidf, # NOT NEEDED IN FEATURE DATA
        tfidfVec,
        # headlineVec, articleVec,
        wordVec,
        # headlineSent,
        # articleSent,
        sentVec,
        svd
    )

    features = np.hstack(features)
    print(features.shape)
    target_data = data['target'].values
    body_ids = data['Body ID'].values

    return [features, target_data, body_ids]
