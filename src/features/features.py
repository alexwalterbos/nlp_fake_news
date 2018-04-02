from .sentiment import generate_sentiment_analysis_files
from .count import generate_count_feature
from .tfidf import generate_tfidf_feature
from .word2vec import generate_word2vec_feature
from .svd import generate_svd_feature


def extract_features(data):
    count = generate_count_feature(data)
    [h_tfidf_train, a_tfidf_train, h_tfidf_test, a_tfidf_test] = generate_tfidf_feature(data)
    word2vec = generate_word2vec_feature(data)
    sentiment = generate_sentiment_analysis_files(data)
    svd = generate_svd_feature(h_tfidf_train, a_tfidf_train, h_tfidf_test, a_tfidf_test)


    print('done')
