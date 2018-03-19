from preprocessing.preprocess import preprocess
from features import features

if __name__ == '__main__':
    data = preprocess()
    print(data.shape)
    extractedFeatures = features.extract_features(data)
