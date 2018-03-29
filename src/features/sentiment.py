import pandas as pd
import numpy as np
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

sent_analyzer = SentimentIntensityAnalyzer()


def compute_sentiment(sentences):
    """
    This computes, per sentence in sentences, the sentiment intensity, using the nltk SentimentIntensityAnalyzer
    :param sentences: A set of sentences from a text.
    :return: the mean sentiment intensity score
    """
    result = []
    for sentence in sentences:
        sentiment_strength = sent_analyzer.polarity_scores(sentence)
        result.append(sentiment_strength)
    return pd.DataFrame(result).mean()


def generate_sentiment_analysis_files(data):

    print('Generating sentiment features')

    # Number of entries in `data` that have a 'target' value
    training_set_size = data[~data['target'].isnull()].shape[0]
    # Number of entries in `data` that have no 'target' value
    test_set_size = data[data['target'].isnull()].shape[0]

    # Headlines are tokenized
    data['headline_sents'] = data['Headline'].apply(lambda headline: sent_tokenize(headline))
    # Apply the sentiment computation to all headline tokens
    data = pd.concat([data, data['headline_sents'].apply(lambda headline: compute_sentiment(headline))], axis=1)
    # Rename outputted columns to prevent conflict with articleBody sentiment columns
    data.rename(columns = {'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)

    headline_sentiment = data[['h_compound', 'h_neg', 'h_neu', 'h_pos']].values
    print('headline_sentiment.shape {}'.format(headline_sentiment.shape))

    # Extract the sentiment values for the training set
    headline_sentiment_train = headline_sentiment[:training_set_size, :]
    # Write it to a file
    with open('feature_sentiment/train.headline.sentiment.pkl', 'wb') as outfile:
        pickle.dump(headline_sentiment_train, outfile, -1)
        print('Headline sentiment features for training set saved')
    # Same for the test set, if available
    if test_set_size > 0:
        headline_sentiment_test = headline_sentiment[test_set_size:, :]
        with open('feature_sentiment/test.headline.sentiment.pkl', 'wb') as outfile:
            pickle.dump(headline_sentiment_test, outfile, -1)
        print('Headline sentiment features for test set saved')
    print('Headline sentiment done')

    # Tokenize bodies
    data['body_sents'] = data['articleBody'].map(lambda body: sent_tokenize(body))
    # Compute sentiment for bodies
    data = pd.concat([data, data['body_sents'].apply(lambda body: compute_sentiment(body))], axis=1)
    # Rename columns to mark them as body sentiment values
    data.rename(columns = {'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)

    body_sentiment = data[['b_compound', 'b_neg', 'b_neu', 'b_pos']].values
    print('body_sentiment.shape {}'.format(body_sentiment.shape))

    # Write the training set to file
    with open('feature_sentiment/train.body.sentiment.pkl', 'wb') as outfile:
        pickle.dump(body_sentiment[:training_set_size, :], outfile, -1)
        print('Body sentiment features for training set saved')

    # Same for test, if applicable
    if test_set_size > 0:
        with open('feature_sentiment/test.body.sentiment.pkl', 'wb') as outfile:
            pickle.dump(body_sentiment[test_set_size:, :], outfile, -1)
            print('Body sentiment features for test set saved')

    print('Body sentiment done')

    return [headline_sentiment, body_sentiment]

