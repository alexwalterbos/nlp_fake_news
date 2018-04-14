import sys

from features.features import extract_features
from preprocessing.preprocess import load_and_preprocess


def main(readfile, limit):
    data = load_and_preprocess(readfile, limit)
    extractedFeatures = extract_features(data)


if __name__ == '__main__':
    readfile = None
    limit = None
    shouldReadFromFile = len(sys.argv) > 0 and 'read' in sys.argv
    if shouldReadFromFile:
        readfile = sys.argv[sys.argv.index('read') + 1]
    if len(sys.argv) > 0 and 'limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('limit') + 1])
    main(readfile, limit)
