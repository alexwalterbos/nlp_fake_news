def magicallyGetData():
    raise NotImplemented


def preprocess(data):
    raise NotImplemented


def generateNGrams(data):
    raise NotImplemented


if __name__ == '__main__':
    # If no 'data.pkl' is present, load data and preprocess:
    data = magicallyGetData()
    prepped = preprocess(data)
    result = generateNGrams(prepped)
    # Else: fetch from file.
