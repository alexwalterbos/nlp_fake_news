import cPickle as cp
import os.path
import sys

import pandas as pd

targets = ['agree', 'disagree', 'discuss', 'unrelated']
targets = dict(zip(targets, range(len(targets))))


def read_data_from_fnc_files():
    print 'Reading data from FNC-provided csv files'
    train_bodies = pd.read_csv('train/train_bodies.csv', encoding='utf-8')
    train_stances = pd.read_csv('train/train_stances.csv', encoding='utf-8')
    training_set = pd.merge(train_stances, train_bodies, how='left', on='Body ID')
    training_set['target'] = map(lambda x: targets[x], training_set['Stance'])
    print 'Shape of training set: ' + str(training_set.shape)
    return training_set


def read_data_from_premade_file():
    filename = sys.argv[sys.argv.index('read') + 1]
    if not os.path.isfile(filename):
        print 'Cannot read from non-existent file: ' + filename
        return None
    else:
        print 'Reading from file: ' + filename
        with open(filename, 'rb') as f:
            data = cp.load(f)
        print 'Shape of loaded training set: ' + str(data.shape)
    return data


def preprocess(data):
    raise NotImplemented


def generate_ngrams(data):
    raise NotImplemented


def obtain_training_data():
    if len(sys.argv) > 0 and 'read' in sys.argv:
        read_data = read_data_from_premade_file()
    else:
        read_data = read_data_from_fnc_files()

    return read_data


if __name__ == '__main__':
    data = obtain_training_data()
    prepped = preprocess(data)
    result = generate_ngrams(prepped)
    # Else: fetch from file.
