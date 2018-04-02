def test_set_size(data: object) -> object:
    return data[data['target'].isnull()].shape[0]


def train_set_size(data):
    return data[~data['target'].isnull()].shape[0]
