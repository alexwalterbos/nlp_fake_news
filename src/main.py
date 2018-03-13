from preprocessing import preprocess
from features import features

if __name__ == '__main__':
    data = preprocess()
    extractedFeatures = features.extract_features(data)
