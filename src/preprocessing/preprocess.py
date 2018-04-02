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

word_pattern = r"(?u)\b\w\w+\b"


def read_data_from_fnc_files(limit):
    # Read data from FNC-provided csv files.
    print('Reading data from FNC-provided csv files')
    train_bodies = pd.read_csv('train/train_bodies.csv', encoding='utf-8')
    train_stances = pd.read_csv('train/train_stances.csv', encoding='utf-8')

    # Generate the training set by merging stances and bodies into one DataFrame
    training_set = pd.merge(train_stances, train_bodies, how='left', on='Body ID')
    # Map the existing stance information from a string representation to a numeric one.
    training_set['target'] = map(lambda x: targets[x], training_set['Stance'])
    print('Shape of training set: ' + str(training_set.shape))

    # Load test data from FNC-provided csv files.
    test_bodies = pd.read_csv('test/test_bodies.csv', encoding='utf-8')
    test_headlines = pd.read_csv('test/test_stances_unlabeled.csv', encoding='utf-8')
    # Generate the test set by merging stances and bodies into one DataFrame
    test_set = pd.merge(test_headlines, test_bodies, how='left', on='Body ID')

    # If the command line argument is given, crop the test- and training set to the provided number of entries
    if limit is not None:
        training_set = training_set[:limit]
        print('Training set trimmed to: ' + str(training_set.shape))

        test_set = test_set[:limit]
        print('Test set trimmed to: ' + str(test_set.shape))

    # Put the training- and test set in the same DataFrame
    data = pd.concat((training_set, test_set))
    print('Shape of data set: {}'.format(data.shape))

    return data


def read_data_from_premade_file(readfile):
    # Check if it exists
    if not os.path.isfile(readfile):
        print('Cannot read from non-existent file: ' + readfile)
        return None
    else:
        print('Reading from file: ' + readfile)
        with open(readfile, 'rb') as f:
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
    # Check if there are enough words to generage an n-gram
    if wordcount >= n:
        # the max `i` must be equal to `wordcount - (n - 1)` because at that index i + n == wordcount - 1, and that's the last index of the list.
        for i in range(wordcount - (n - 1)):
            words_in_gram = []
            # Get all words that must go into this n-gram and put them in a temporary list `words_in_gram`
            for k in range(0, n):
                words_in_gram.append(words[i + k])
            # put the n-gram in the list of n-grams for the considered words
            grams.append('_'.join(words_in_gram))
    else:
        # Fallback to an '(n-1)-gram' because there weren't enough words to make an n-gram
        print('Word count was too low for {}-gram, making {}-gram'.format(n, n - 1))
        grams = generate_ngram(words, n - 1)
    return grams


def extract_tokens(
        line,
        word_pattern=word_pattern
):
    # Compile a pattern based on `word_pattern` which is a regex that filters out entire words.
    pattern = re.compile(word_pattern)
    # Apply a stemmer to all tokens that match the pattern
    stemmed_tokens = [stemmer.stem(x.lower()) for x in pattern.findall(line)]
    # Filter out stopwords to retain only useful words.
    filtered_stemmed_tokens = [x for x in stemmed_tokens if x not in stopwords]

    return filtered_stemmed_tokens


def preprocess(readfile, limit):
    # Check for read argument, and load data from provided file
    if readfile is not None:
        data = read_data_from_premade_file(readfile)
    else:
        # Read data from FNC-provided files
        read_data = read_data_from_fnc_files(limit)
        data = append_ngrams(read_data)
        print(data.axes)

        print('Successfully loaded data!')

    return data
