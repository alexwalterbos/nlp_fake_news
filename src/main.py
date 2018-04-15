import sys
import pickle
import os.path
import dill as pickle
import nltk

from features.features import extract_features
from preprocessing.preprocess import load_and_preprocess


def main(limit):
    # Install required nltk resources
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Generate filenames based on the provided limit argument
    limit_suffix = '_limited_' + str(limit) if limit is not None else ''
    feature_filename = 'data_files/feature_data' + limit_suffix + '.pkl'
    target_filename = 'data_files/target_data' + limit_suffix + '.pkl'
    body_ids_filename = 'data_files/body_ids_data' + limit_suffix + '.pkl'

    # Check if files have been generated for this limit, or generate new feature data if not
    if not (os.path.isfile(feature_filename) and os.path.isfile(target_filename) and os.path.isfile(body_ids_filename)):
        data = load_and_preprocess(limit)
        [feature_data, target_data, body_ids] = extract_features(data)
        with open(feature_filename, 'wb') as feature_file:
            pickle.dump(feature_data, feature_file, -1)
            print('Saved feature data')

        with open(target_filename, 'wb') as target_file:
            pickle.dump(target_data, target_file, -1)
            print('Saved target data')

        with open(body_ids_filename, 'wb') as body_ids_file:
            pickle.dump(body_ids, body_ids_file, -1)
            print('Saved body_ids')
    else:
        with open(feature_filename, 'rb') as feature_file:
            feature_data = pickle.load(feature_file)
        with open(target_filename, 'rb') as target_file:
            target_data = pickle.load(target_file)
        with open(body_ids_filename, 'rb') as body_ids_file:
            body_ids = pickle.load(body_ids_file)

    print(feature_data.shape)
    print(target_data.shape)
    print(body_ids.shape)
    # We now have the feature data, target data and body ids ready for use in XGB.



if __name__ == '__main__':
    limit = None
    if len(sys.argv) > 0 and 'limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('limit') + 1])
    main(limit)
