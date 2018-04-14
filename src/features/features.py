import pickle
from features.sentiment import generate_sentiment_analysis_files
from features.count import generate_count_feature
from features.tfidf import generate_tfidf_feature
from features.word2vec import generate_word2vec_feature
from features.svd import generate_svd_feature
import numpy
from util import test_set_size, train_set_size


def extract_features(data):
    '''
    print('Generating count feature')
    count = generate_count_feature(data)

    print('Generating tfidf feature')
    [headlineTfidf, bodyTfidf, tfidfVec] = generate_tfidf_feature(data)
    '''
    print('Generating word2vec feature')
    [headlineVec, articleVec, simVec] = generate_word2vec_feature(data,train_set_size(data))

    '''
    print('Generating sentiment feature')
    [headlineSent, articleSent] = generate_sentiment_analysis_files(data)

    print('Generating svd feature')
    svd = generate_svd_feature(
        headlineTfidf,
        bodyTfidf,
        test_set_size(data),
        train_set_size(data)
    )
    '''

    with open('features/feature_pickles/train.pkl', "wb") as outfile:
        pickle.dump(simVec[:train_set_size(data)], outfile, -1)

    with open('features/feature_pickles/test.pkl', "wb") as outfile:
        pickle.dump(simVec[train_set_size(data):], outfile, -1)

    features = [[
        count,
        headlineTfidf,
        bodyTfidf,
        tfidfVec,
        headlineVec,
        articleVec,
        wordVec,
        headlineSent,
        articleSent,
        svd
    ]]

    feature_data = numpy.hstack(features)
    target_data = data['target'].values
    body_ids = data['Body ID'].values

    return [feature_data, target_data, body_ids]
