import sys
import pickle
import numpy as np
from itertools import chain
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GroupKFold

import xgboost as xgb
from collections import Counter
import count
#import tfidf
import sentiment
import svd
import word2vec
from score import *            #TODO
import pandas as pd
'''
    10-fold cv on 80% of the data (training_ids.txt)
    splitting based on BodyID
    test on remaining 20% (hold_out_ids.txt)
'''

params_xgb = {

    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    #'objective': 'multi:softmax',
    'objective': 'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class': 4
}

num_round = 1000

def read(header='train'):
    filename_simvec = "feature_pickles/%s.pkl" % header
    with open(filename_simvec, "rb") as infile:
        simVec = pickle.load(infile)
    return simVec

def build_data():

    # create target variable
    body = pd.read_csv("../../train/train_bodies.csv")
    stances = pd.read_csv("../../train/train_stances.csv")
    data = pd.merge(stances, body, how='left', on='Body ID')
    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = map(lambda x: targets_dict[x], data['Stance'])

    data_y = data['target'].values
    print(data_y)
    true_data_y = []
    print()
    for z in data_y:
        for z2 in z:
            true_data_y.append(z2)
    #print(true_data_y)

    # read features
    generators = [
        #count,
#       tfidf,
        #svd,
        word2vec
        #sentiment
    ]
    featuresVar = []
    for f in read():
        featuresVar.append([f])
    data_x = np.array(featuresVar)

    #print(data_x)
    #print("")
    print ("train = ")
    print(data_x.shape)

    #print (data_x[0,:])
    # print 'data_x.shape'
    # print data_x.shape
    # print 'data_y.shape'
    # print data_y.shape
    # print 'body_ids.shape'
    # print data['Body ID'].values.shape

    # with open('data_new.pkl', 'wb') as outfile:
    #    pickle.dump(data_x, outfile, -1)
    #    #print 'data saved in data_new.pkl'

    return data_x, true_data_y, data['Body ID'].values

def build_test_data():
    # create target variable
    # replace file names when test data is ready
    body = pd.read_csv("../../test/test_bodies.csv")
    stances = pd.read_csv("../../test/test_stances_unlabeled.csv")  # needs to contain pair id
    data = pd.merge(stances, body, how='left', on='Body ID')

    # read feavures
    generators = [
        #count,
        #tfidf(),
        #svd,
        word2vec,
        #sentiment
    ]

    featuresVar = []
    for f in read("test"):
        featuresVar.append([f])
    data_x = np.array(featuresVar)

    print("test = ")
    print(data_x.shape)
    # print data_x[0,:]
    # print 'test data_x.shape'
    # print data_x.shape
    # print 'test body_ids.shape'
    # print data['Body ID'].values.shape
    # pair id
    return data_x, data['Body ID'].values


def eval_metric(yhat, dtrain):
    y = dtrain.get_label()
    yhat = np.argmax(yhat, axis=1)
    predicted = [LABELS[int(a)] for a in yhat]
    actual = [LABELS[int(a)] for a in y]
    s, _ = score_submission(actual, predicted)
    s_perf, _ = score_submission(actual, actual)
    score = float(s) / s_perf
    return 'score', score


def train():
    print("test")
    data_x, data_y, body_ids = build_data()
    # read test data
    test_x, body_ids_test = build_test_data()

    w = np.array([1 if y == 3 else 4 for y in data_y])
    print ('w:')
    print (w)
    print (np.mean(w))

    n_iters = 500
    # n_iters = 50
    # perfect score on training set
    # print 'perfect_score: ', perfect_score(data_y)
    # print Counter(data_y)

    print(data_x.size)
    print(len(data_y))
    print()
    #data_x = [1, 2, 3]
    #data_y = [2, 3, 4]
    #w = [4, 4, 4]

    dtrain = xgb.DMatrix(data_x, label=data_y, weight=w)
    dtest = xgb.DMatrix(test_x)
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params_xgb,
                    dtrain,
                    n_iters,
                    watchlist,
                    feval=eval_metric,
                    verbose_eval=10)

    # pred_y = bst.predict(dtest) # output: label, not probabilities
    # pred_y = bst.predict(dtrain) # output: label, not probabilities
    pred_prob_y = bst.predict(dtest).reshape(test_x.shape[0], 4)  # predicted probabilities
    pred_y = np.argmax(pred_prob_y, axis=1)
    # print 'pred_y.shape:'
    # print pred_y.shape
    predicted = [LABELS[int(a)] for a in pred_y]
    print(len(predicted))
    print()

    # save (id, predicted and probabilities) to csv, for model averaging
    #stances = pd.read_csv("test_stances_unlabeled_processed.csv") # same row order as predicted
    stances = pd.read_csv("../../test/test_stances_unlabeled.csv")  # same row order as predicted
    print(stances.shape)
    #TODO

    df_output = pd.DataFrame()
    df_output['Headline'] = stances['Headline']
    df_output['Body ID'] = stances['Body ID']
    df_output['Stance'] = predicted
    df_output['prob_0'] = pred_prob_y[:, 0]
    df_output['prob_1'] = pred_prob_y[:, 1]
    df_output['prob_2'] = pred_prob_y[:, 2]
    df_output['prob_3'] = pred_prob_y[:, 3]
    # df_output.to_csv('submission.csv', index=False)
    df_output.to_csv('tree_pred_prob_cor2.csv', index=False)
    df_output[['Headline', 'Body ID', 'Stance']].to_csv('tree_pred_cor2.csv', index=False)

    print (df_output)
    # print Counter(df_output['Stance'])

    # pred_train = bst.predict(dtrain).reshape(data_x.shape[0], 4)
    # pred_t = np.argmax(pred_train, axis=1)
    # predicted_t = [LABELS[int(a)] for a in pred_t]
    ##print Counter(predicted_t)


def cv():
    data_x, data_y, body_ids = build_data()

    holdout_ids = set([int(x.rstrip()) for x in file('hold_out_ids.txt')])
    # print 'len(holdout_ids): ',len(holdout_ids)
    holdout_idx = [t for (t, x) in enumerate(body_ids) if x in holdout_ids]
    test_x = data_x[holdout_idx]  # features of test set
    # print 'holdout_x.shape: '
    # print test_x.shape
    test_y = data_y[holdout_idx]
    # print Counter(test_y)
    # return 1

    # to obtain test dataframe for model averaging
    body = pd.read_csv("train_bodies.csv")
    stances = pd.read_csv("train_stances.csv")
    data = pd.merge(stances, body, how='left', on='Body ID')
    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = map(lambda x: targets_dict[x], data['Stance'])
    test_df = data.ix[holdout_idx]

    cv_ids = set([int(x.rstrip()) for x in file('training_ids.txt')])
    # print 'len(cv_ids): ',len(cv_ids)
    cv_idx = [t for (t, x) in enumerate(body_ids) if x in cv_ids]
    cv_x = data_x[cv_idx]
    # print 'cv_x.shape: '
    # print cv_x.shape
    cv_y = data_y[cv_idx]
    groups = body_ids[cv_idx]  # GroupKFold will make sure all samples
    # having the same "Body ID" will appear in the same fold
    w = np.array([1 if y == 3 else 4 for y in cv_y])
    # print 'w:'
    # print w
    # print np.mean(w)

    scores = []
    wscores = []
    pscores = []
    n_folds = 10
    best_iters = [0] * n_folds
    kf = GroupKFold(n_splits=n_folds)
    # need to create disjoint sets for training and validation
    for fold, (trainInd, validInd) in enumerate(kf.split(cv_x, cv_y, groups)):
        continue
        # print 'fold %s' % fold
        x_train = cv_x[trainInd]
        y_train = cv_y[trainInd]
        x_valid = cv_x[validInd]
        y_valid = cv_y[validInd]
        idx_valid = np.array(cv_idx)[validInd]

        # print 'perfect_score: ', perfect_score(y_valid)
        # print Counter(y_valid)
        # break
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=w[trainInd])
        dvalid = xgb.DMatrix(x_valid, label=y_valid, weight=w[validInd])
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(params_xgb,
                        dtrain,
                        num_round,
                        watchlist,
                        verbose_eval=10,
                        # feval = eval_metric,
                        # maximize = True,
                        early_stopping_rounds=80)

        # pred_y = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)
        pred_y = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit).reshape(y_valid.shape[0], 4)
        # print 'predicted probabilities: '
        # print pred_y
        pred_y = np.argmax(pred_y, axis=1)
        # print 'predicted label indices: '
        # print pred_y

        # print 'best iterations: ', bst.best_ntree_limit
        best_iters[fold] = bst.best_ntree_limit

        # pred_y = bst.predict(dvalid)
        # print pred_y
        ##print Counter(pred_y)
        # pred_y = np.argmax(bst.predict(dvalid, ntree_limit=bst.best_ntree_limit), axis=1)
        # print 'pred_y.shape'
        # print pred_y.shape
        # print 'y_valid.shape'
        # print y_valid.shape
        # s = fscore(pred_y, y_valid)
        # s_perf = perfect_score(y_valid)
        predicted = [LABELS[int(a)] for a in pred_y]
        actual = [LABELS[int(a)] for a in y_valid]
        # #print out the headline & body text for incorrect predictions
        # show_incorrect_pred(actual, predicted, idx_valid)

        s, _ = score_submission(actual, predicted)
        s_perf, _ = score_submission(actual, actual)
        wscore = float(s) / s_perf
        # print 'fold %s, score = %f, perfect_score %f, weighted percentage %f' % (fold, s, s_perf, wscore)
        scores.append(s)
        pscores.append(s_perf)
        wscores.append(wscore)
        # break

    # print 'scores:'
    # print scores
    # print 'mean score:'
    # print np.mean(scores)
    # print 'perfect scores:'
    # print pscores
    # print 'mean perfect score:'
    # print np.mean(pscores)
    # print 'w scores:'
    # print wscores
    # print 'mean w score:'
    # print np.mean(wscores)
    # print 'best iters:'
    # print best_iters
    # print 'mean best_iter:'
    m_best = np.mean(best_iters)
    # print m_best
    # m_best = best_iters[0]
    m_best = 500
    # m_best = 600
    # return 1

    # use the same parameters to train with full cv data, test on hold-out data
    # print 'test on holdout set'
    dtrain = xgb.DMatrix(cv_x, label=cv_y, weight=w)
    dtest = xgb.DMatrix(test_x, label=test_y)
    watchlist = [(dtrain, 'train')]
    clf = xgb.train(params_xgb,
                    dtrain,
                    # num_round,
                    int(m_best),
                    watchlist,
                    feval=eval_metric,
                    verbose_eval=10)

    pred_prob_holdout_y = clf.predict(dtest).reshape(test_y.shape[0], 4)  # probabilities
    pred_holdout_y = np.argmax(pred_prob_holdout_y, axis=1)
    # print 'pred_holdout_y.shape:'
    # print pred_holdout_y.shape
    # print 'test_y.shape:'
    # print test_y.shape
    # s_test = fscore(pred_holdout_y, test_y)
    # s_test_perf = perfect_score(test_y)
    predicted = [LABELS[int(a)] for a in pred_holdout_y]
    actual = [LABELS[int(a)] for a in test_y]
    report_score(actual, predicted)
    # print Counter(predicted)

    test_df['actual'] = actual
    test_df['predicted'] = predicted
    test_df['prob_0'] = pred_prob_holdout_y[:, 0]
    test_df['prob_1'] = pred_prob_holdout_y[:, 1]
    test_df['prob_2'] = pred_prob_holdout_y[:, 2]
    test_df['prob_3'] = pred_prob_holdout_y[:, 3]

    # test_df[['Headline','Body ID', 'Stance', 'actual', 'predicted']].to_csv('predtest.csv', index=False)
    test_df[['Headline', 'Body ID', 'Stance', 'actual', 'predicted', 'prob_0', 'prob_1', 'prob_2', 'prob_3']].to_csv(
        'predtest_cor2.csv', index=False)

    ##print 'on holdout set, score = %f, perfect_score %f' % (s_test, s_test_perf)

if __name__ == "__main__":
    print(" hello world!")
    stances = pd.read_csv(
        "../../train/train_stances.csv")  # same row order as predicted
    print(stances.shape)
    stances = pd.read_csv(
        "../../competition_test/competition_test_stances.csv")  # same row order as predicted
    print(stances.shape)
    # TODO

    #df_output = pd.DataFrame()
    #df_output['Headline'] = stances['Headline']
    #df_output['Body ID'] = stances['Body ID']

    train()