import pickle

import pandas as pd
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
    # Headlines are tokenized
    data['headline_sents'] = data['Headline'].apply(lambda headline: sent_tokenize(headline))
    # Apply the sentiment computation to all headline tokens
    data = pd.concat([data, data['headline_sents'].apply(lambda headline: compute_sentiment(headline))], axis=1)
    # Rename outputted columns to prevent conflict with articleBody sentiment columns
    data.rename(columns={'compound': 'h_compound', 'neg': 'h_neg', 'neu': 'h_neu', 'pos': 'h_pos'}, inplace=True)

    headline_sentiment = data[['h_compound', 'h_neg', 'h_neu', 'h_pos']].values

    # Tokenize bodies
    data['body_sents'] = data['articleBody'].map(lambda body: sent_tokenize(body))
    # Compute sentiment for bodies
    data = pd.concat([data, data['body_sents'].apply(lambda body: compute_sentiment(body))], axis=1)
    # Rename columns to mark them as body sentiment values
    data.rename(columns={'compound': 'b_compound', 'neg': 'b_neg', 'neu': 'b_neu', 'pos': 'b_pos'}, inplace=True)

    body_sentiment = data[['b_compound', 'b_neg', 'b_neu', 'b_pos']].values

    return [headline_sentiment, body_sentiment]
