import _pickle as pickle
import os.path
import re
import sys

import nltk
import pandas as pd

stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))
targets = ['agree', 'disagree', 'discuss', 'unrelated']
targets = dict(zip(targets, range(len(targets))))

token_pattern = r"(?u)\b\w\w+\b"


def read_data_from_fnc_files():
    print('Reading data from FNC-provided csv files')

    train_bodies = pd.read_csv('train/train_bodies.csv', encoding='utf-8')
    train_stances = pd.read_csv('train/train_stances.csv', encoding='utf-8')
    training_set = pd.merge(train_stances, train_bodies, how='left', on='Body ID')
    training_set['target'] = map(lambda x: targets[x], training_set['Stance'])
    print('Shape of training set: ' + str(training_set.shape))

    test_bodies = pd.read_csv('test/test_bodies.csv', encoding='utf-8')
    test_headlines = pd.read_csv('test/test_stances_unlabeled.csv', encoding='utf-8')
    test_set = pd.merge(test_headlines, test_bodies, how='left', on='Body ID')

    if len(sys.argv) > 0 and 'limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('limit') + 1])
        training_set = training_set[:limit]
        print('Training set trimmed by system argument to: ' + str(training_set.shape))

        test_set = test_set[:limit]
        print('Test set trimmed by system argument to: ' + str(test_set.shape))

    data = pd.concat((training_set, test_set))
    print('Shape of data set: {}'.format(data.shape))

    return data


def read_data_from_premade_file():
    filename = sys.argv[sys.argv.index('read') + 1]
    if not os.path.isfile(filename):
        print('Cannot read from non-existent file: ' + filename)
        return None
    else:
        print('Reading from file: ' + filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print('Shape of loaded training set: ' + str(data.shape))
    return data


def append_ngrams(data):
    print('Generating unigram for headlines')
    headline_unigrams = data['Headline'].map(lambda headline: extract_tokens(headline))
    print('Generating unigram for bodies')
    body_unigrams = data['articleBody'].map(lambda body: extract_tokens(body))
    print('Generating bigram for headlines')
    headline_bigrams = headline_unigrams.map(lambda headline: generate_ngram(headline, 2))
    print('Generating bigram for bodies')
    body_bigrams = body_unigrams.map(lambda body: generate_ngram(body, 2))
    print('Generating trigram for headlines')
    headline_trigrams = headline_unigrams.map(lambda headline: generate_ngram(headline, 3))
    print('Generating trigram for bodies')
    body_trigrams = body_unigrams.map(lambda body: generate_ngram(body, 3))

    data['Headline_unigram'] = headline_unigrams
    data['articleBody_unigram'] = body_unigrams
    data['Headline_bigram'] = headline_bigrams
    data['articleBody_bigram'] = body_bigrams
    data['Headline_trigram'] = headline_trigrams
    data['articleBody_trigram'] = body_trigrams

    return data


def generate_ngram(words, n):
    wordcount = len(words)
    grams = []
    if wordcount >= n:
        for i in range(wordcount - (n - 1)):
            words_in_gram = []
            for k in range(0, n):
                words_in_gram.append(words[i + k])
            grams.append('_'.join(words_in_gram))
    else:
        print('Word count was too low for {}-gram, making {}-gram'.format(n, n - 1))
        grams = generate_ngram(words, n - 1)
    return grams


def extract_tokens(
        line,
        token_pattern=token_pattern
):
    pattern = re.compile(token_pattern, flags=re.UNICODE | re.LOCALE)
    stemmed_tokens = [stemmer.stem(x.lower()) for x in pattern.findall(line)]
    filtered_stemmed_tokens = [x for x in stemmed_tokens if x not in stopwords]

    return filtered_stemmed_tokens


def preprocess():
    if len(sys.argv) > 0 and 'read' in sys.argv:
        data = read_data_from_premade_file()
    else:
        read_data = read_data_from_fnc_files()
        data = append_ngrams(read_data)
        print(data.axes)

        print('Success!')

    return data
