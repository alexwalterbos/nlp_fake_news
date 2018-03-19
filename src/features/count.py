import pickle as cp
import pandas as pd

feat_names = ["Headline", "articleBody"]	

with open('smalldata.pkl', 'rb') as file:
	data = cp.load(file, encoding="latin1") 
	
countFeature = {}
countFeature["count_Headline_unigram"] = list(data.apply(lambda x: len(x["Headline_unigram"]), axis=1))
countFeature["count_unique_Headline_unigram"] = list(data.apply(lambda x: len(set(x["Headline_unigram"])), axis=1))
countFeature["ratio_unique_Headline_unigram"] = [x/y for x,y in zip(countFeature["count_Headline_unigram"],countFeature["count_unique_Headline_unigram"])]	
	
countFeature["count_Headline_bigram"] = list(data.apply(lambda x: len(x["Headline_bigram"]), axis=1))
countFeature["count_unique_Headline_bigram"] = list(data.apply(lambda x: len(set(x["Headline_bigram"])), axis=1))
countFeature["ratio_unique_Headline_bigram"] = [x/y for x,y in zip(countFeature["count_Headline_bigram"],countFeature["count_unique_Headline_bigram"])]	

countFeature["count_Headline_trigram"] = list(data.apply(lambda x: len(x["Headline_trigram"]), axis=1))
countFeature["count_unique_Headline_trigram"] = list(data.apply(lambda x: len(set(x["Headline_trigram"])), axis=1))
countFeature["ratio_unique_Headline_trigram"] = [x/y for x,y in zip(countFeature["count_Headline_trigram"],countFeature["count_unique_Headline_trigram"])]	
	
countFeature["count_articleBody_unigram"] = list(data.apply(lambda x: len(x["articleBody_unigram"]), axis=1))
countFeature["count_unique_articleBody_unigram"] = list(data.apply(lambda x: len(set(x["articleBody_unigram"])), axis=1))
countFeature["ratio_unique_articleBody_unigram"] = [x/y for x,y in zip(countFeature["count_articleBody_unigram"],countFeature["count_unique_articleBody_unigram"])]	
	
countFeature["count_articleBody_bigram"] = list(data.apply(lambda x: len(x["articleBody_bigram"]), axis=1))
countFeature["count_unique_articleBody_bigram"] = list(data.apply(lambda x: len(set(x["articleBody_bigram"])), axis=1))
countFeature["ratio_unique_articleBody_bigram"] = [x/y for x,y in zip(countFeature["count_articleBody_bigram"],countFeature["count_unique_articleBody_bigram"])]	
	
countFeature["count_articleBody_trigram"] = list(data.apply(lambda x: len(x["articleBody_trigram"]), axis=1))
countFeature["count_unique_articleBody_trigram"] = list(data.apply(lambda x: len(set(x["articleBody_trigram"])), axis=1))
countFeature["ratio_unique_articleBody_trigram"] = [x/y for x,y in zip(countFeature["count_articleBody_trigram"],countFeature["count_unique_articleBody_trigram"])]		
	
countFeature["count_Headline_unigram_in_articleBody"] = list(data.apply(lambda x: sum([1. for w in x["Headline_unigram"] if w in set(x["articleBody_unigram"])]), axis=1)) 	
countFeature["count_Headline_bigram_in_articleBody"] = list(data.apply(lambda x: sum([1. for w in x["Headline_bigram"] if w in set(x["articleBody_bigram"])]), axis=1)) 	
countFeature["count_Headline_trigram_in_articleBody"] = list(data.apply(lambda x: sum([1. for w in x["Headline_trigram"] if w in set(x["articleBody_trigram"])]), axis=1))

countFeature["ratio_Headline_unigram_in_articleBody"] = [x/y for x,y in zip(countFeature["count_Headline_unigram_in_articleBody"],countFeature["count_Headline_unigram"])]	 	
countFeature["ratio_Headline_bigram_in_articleBody"] = [x/y for x,y in zip(countFeature["count_Headline_bigram_in_articleBody"],countFeature["count_Headline_bigram"])]	
countFeature["ratio_Headline_trigram_in_articleBody"] = [x/y for x,y in zip(countFeature["count_Headline_trigram_in_articleBody"],countFeature["count_Headline_trigram"])]
	
countsTrain = data[feat_names].values
outfilename_train = "feature_pickles/count.train.pkl"
with open(outfilename_train, "wb") as outfile:
    cp.dump(feat_names, outfile, -1)
    cp.dump(countsTrain, outfile, -1)
	
if data.shape[0] > 0:
    countsTest = data[feat_names].values
    outfilename_test = "feature_pickles/count.test.pkl"
    with open(outfilename_test, 'wb') as outfile:
        cp.dump(feat_names, outfile, -1)
        cp.dump(countsTest, outfile, -1)

	
	