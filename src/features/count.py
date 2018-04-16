import pandas as pd

def generate_count_feature(data):
    feat_names = ["Headline", "articleBody"]
    countFeature = pd.DataFrame()

    # Calculation of the count, unique count and ratio between these two for the Headline unigram
    countFeature["count_Headline_unigram"] = list(data.apply(lambda x: len(x["Headline_unigram"]), axis=1))
    countFeature["count_unique_Headline_unigram"] = list(data.apply(lambda x: len(set(x["Headline_unigram"])), axis=1))
    countFeature["ratio_unique_Headline_unigram"] = [x / y for x, y in zip(countFeature["count_Headline_unigram"],
                                                                           countFeature[
                                                                               "count_unique_Headline_unigram"])]

    # Calculation of the count, unique count and ratio between these two for the Headline bigram
    countFeature["count_Headline_bigram"] = list(data.apply(lambda x: len(x["Headline_bigram"]), axis=1))
    countFeature["count_unique_Headline_bigram"] = list(data.apply(lambda x: len(set(x["Headline_bigram"])), axis=1))
    countFeature["ratio_unique_Headline_bigram"] = [x / y for x, y in zip(countFeature["count_Headline_bigram"],
                                                                          countFeature["count_unique_Headline_bigram"])]

    # Calculation of the count, unique count and ratio between these two for the Headline trigram
    countFeature["count_Headline_trigram"] = list(data.apply(lambda x: len(x["Headline_trigram"]), axis=1))
    countFeature["count_unique_Headline_trigram"] = list(data.apply(lambda x: len(set(x["Headline_trigram"])), axis=1))
    countFeature["ratio_unique_Headline_trigram"] = [x / y for x, y in zip(countFeature["count_Headline_trigram"],
                                                                           countFeature[
                                                                               "count_unique_Headline_trigram"])]

    # Calculation of the count, unique count and ratio between these two for the body unigram
    countFeature["count_articleBody_unigram"] = list(data.apply(lambda x: len(x["articleBody_unigram"]), axis=1))
    countFeature["count_unique_articleBody_unigram"] = list(
        data.apply(lambda x: len(set(x["articleBody_unigram"])), axis=1))
    countFeature["ratio_unique_articleBody_unigram"] = [x / y for x, y in zip(countFeature["count_articleBody_unigram"],
                                                                              countFeature[
                                                                                  "count_unique_articleBody_unigram"])]

    # Calculation of the count, unique count and ratio between these two for the body bigram
    countFeature["count_articleBody_bigram"] = list(data.apply(lambda x: len(x["articleBody_bigram"]), axis=1))
    countFeature["count_unique_articleBody_bigram"] = list(
        data.apply(lambda x: len(set(x["articleBody_bigram"])), axis=1))
    countFeature["ratio_unique_articleBody_bigram"] = [x / y for x, y in zip(countFeature["count_articleBody_bigram"],
                                                                             countFeature[
                                                                                 "count_unique_articleBody_bigram"])]

    # Calculation of the count, unique count and ratio between these two for the body trigram
    countFeature["count_articleBody_trigram"] = list(data.apply(lambda x: len(x["articleBody_trigram"]), axis=1))
    countFeature["count_unique_articleBody_trigram"] = list(
        data.apply(lambda x: len(set(x["articleBody_trigram"])), axis=1))
    countFeature["ratio_unique_articleBody_trigram"] = [x / y for x, y in zip(countFeature["count_articleBody_trigram"],
                                                                              countFeature[
                                                                                  "count_unique_articleBody_trigram"])]

    # Calculation of the number of headline grams appearing in the article body
    countFeature["count_Headline_unigram_in_articleBody"] = list(
        data.apply(lambda x: sum([1. for w in x["Headline_unigram"] if w in set(x["articleBody_unigram"])]), axis=1))
    countFeature["count_Headline_bigram_in_articleBody"] = list(
        data.apply(lambda x: sum([1. for w in x["Headline_bigram"] if w in set(x["articleBody_bigram"])]), axis=1))
    countFeature["count_Headline_trigram_in_articleBody"] = list(
        data.apply(lambda x: sum([1. for w in x["Headline_trigram"] if w in set(x["articleBody_trigram"])]), axis=1))

    # Ratio calculation of the number of headline grams appearing in the article body
    countFeature["ratio_Headline_unigram_in_articleBody"] = [x / y for x, y in
                                                             zip(countFeature["count_Headline_unigram_in_articleBody"],
                                                                 countFeature["count_Headline_unigram"])]
    countFeature["ratio_Headline_bigram_in_articleBody"] = [x / y for x, y in
                                                            zip(countFeature["count_Headline_bigram_in_articleBody"],
                                                                countFeature["count_Headline_bigram"])]
    countFeature["ratio_Headline_trigram_in_articleBody"] = [x / y for x, y in
                                                             zip(countFeature["count_Headline_trigram_in_articleBody"],
                                                                 countFeature["count_Headline_trigram"])]

    return countFeature
