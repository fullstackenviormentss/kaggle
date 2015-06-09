import sys
import csv
import math
import operator
import argparse
import bisect
import random
from pprint import pprint

from collections import defaultdict
from collections import namedtuple
from collections import deque

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pylab as P

from scipy import sparse

from sklearn import svm
from sklearn import tree
from sklearn.externals import joblib

from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, chi2, f_classif, f_regression, SelectFpr
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.pipeline import _name_estimators
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

from mlxtend.sklearn import ColumnSelector
from mlxtend.sklearn import EnsembleClassifier

import shared
import utils
from bidder import Bidder, Interval, Increment, StoredIncrement
from utils import skip, print_X, print_bidders, time_to_hour, divide, get_ip_part
from bids import Bid

import numpy as np
from tabulate import tabulate


class Model:
    def __init__(self):
        # self.features_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
        # self.features_selector = SelectKBest(k="all")
        # self.features_selector = SelectPercentile(score_func=SelectFpr, percentile=16)
        self.features_selector = ExtraTreesClassifier(n_estimators=250,max_features=20)

        self.dict_vectorizer = DictVectorizer()
        self.scaler = StandardScaler(copy=True)

    def vectorize(self, X, y, fit=True):
        # digitize categories
        if fit:
            self.dict_vectorizer.fit(X)
        X = self.dict_vectorizer.transform(X).toarray()
        return X, y

    def scale(self, X, y, fit=True):
        # scale numbers
        if fit:
            self.scaler.fit(X)
        X = self.scaler.transform(X)
        return X, y

    def all_feature_names(self):
        return self.dict_vectorizer.get_feature_names()

    def selected_feature_names(self):
        names = []
        all_names = np.array(self.all_feature_names())
        return all_names[self.feats['ensemble']]

        # if hasattr(self.features_selector, 'get_support'):
        #     for i in self.features_selector.get_support(indices=True):
        #         names.append(all_names[i])
        # else:
        #     feature_importance = self.features_selector.feature_importances_
        #     feature_importance = 100.0 * (feature_importance / feature_importance.max())
        #     sorted_idx = np.argsort(feature_importance)[::-1]
        #     names = np.array(self.all_feature_names()[:len(sorted_idx)])
        #     """
        #     for name, imp in zip(names[sorted_idx], feature_importance[sorted_idx]):
        #         # i = indices[f]
        #         print "%s (%f)" % (name, imp),
        #     """
        # sel_count = int(math.log(len(sorted_idx), 2))
        # return names[sorted_idx][:self.features_selector.max_features]

    def save_features(self, X, y):
        feats = dict()

        print "univariate feature selectors"
        selector_clf = SelectKBest(score_func = f_classif, k = 'all')
        selector_clf.fit(X, y)
        pvalues_clf = selector_clf.pvalues_
        pvalues_clf[np.isnan(pvalues_clf)] = 1

        #put feature vectors into dictionary
        feats['univ_sub01'] = (pvalues_clf<0.1)
        feats['univ_sub005'] = (pvalues_clf<0.05)
        feats['univ_clf_sub005'] = (pvalues_clf<0.05)

        print "randomized logistic regression feature selector"
        sel_log = linear_model.RandomizedLogisticRegression(random_state = 42, n_jobs = 4).fit(X, y)
        #put rand_lasso feats into feature dict
        feats['rand_logreg'] = sel_log.get_support()

        print "l1-based feature selectors"
        X_sp = sparse.coo_matrix(X)
        sel_svc = svm.LinearSVC(C=0.1, penalty = "l1", dual = False, random_state = 42).fit(X, y)
        feats['LinearSVC'] = np.ravel(sel_svc.coef_>0)
        sel_log = linear_model.LogisticRegression(C=0.01, random_state = 42).fit(X_sp, y)
        feats['LogReg'] = np.ravel(sel_log.coef_>0)

        tree_max_features = 20
        print "ExtraTrees feature selectors (%s)" % tree_max_features
        feats['tree'] = np.zeros(len(feats['LogReg']))
        tree = ExtraTreesClassifier(n_estimators=250, max_features=tree_max_features)
        tree.fit(X, y)
        feature_importance = tree.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)[::-1]
        for i in xrange(tree_max_features):
            feats['tree'][sorted_idx[i]] = 1

        feat_sums = np.zeros(len(feats['LogReg']))
        for key in feats:
            feat_sums+=feats[key].astype(int)
        feats['ensemble'] = feat_sums>=4 #take features which get 5 or more votes
        joblib.dump(feats, 'features/feats.pkl', compress = 3)
        return feats

    def load_features(self):
        return joblib.load('features/feats.pkl')

    def select_features(self, X, y, fit=True):
        if fit:
            # self.features_selector.fit(X,y)
            # print "Selected Features:"
            # print self.selected_feature_names()
            # print
            self.feats = self.save_features(X, y)
            # pass
        # self.feats = self.load_features()
        # X = self.features_selector.transform(X)
        print "Selected Features:"
        print self.selected_feature_names()
        print
        return X[:, self.feats['ensemble']], y

    def split_data(self, X, y, ids, cross_validate):
        if not cross_validate:
            return X, [], y, [], ids, []

        # append ids so we can identify who is in test and who is in train set
        X = np.c_[X, ids]
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3) # , random_state=0
        # store ids
        train_ids = X_train[:,-1]
        test_ids = X_test[:,-1]
        # remove ids
        X_train = np.delete(X_train, -1, 1).astype(np.float)
        X_test = np.delete(X_test, -1, 1).astype(np.float)

        return X_train, X_test, y_train, y_test, train_ids, test_ids

    def get_columns_from_selected_features(self, featureNames):
        all_names = self.all_feature_names()
        featureNames = set(featureNames)
        columns = []

        for i,j in enumerate(self.features_selector.get_support(indices=True)):
            if all_names[j] in featureNames:
                columns.append(i)

        return columns

    def get_columns_for_features(self, featureNames):
        all_names = self.all_feature_names()
        cols = []
        for feature in featureNames:
            cols.append(all_names.index(feature))
        return cols

    def standard_prepare(self, X, y, fit=True, cross_validate=True):
        X,y = self.vectorize(X, y, fit)
        X,y = self.select_features(X, y, fit)

        X = np.array(X)
        y = np.array(y)
        self.X_unscaled = X
        self.y_unscaled = y

        X, y = self.scale(X, y, fit)
        self.X_scaled = X
        self.y_scaled = y

        return X, y

    def prepare(self, X, y, ids, fit=True, cross_validate=True):
        X, y = self.standard_prepare(X, y, fit, cross_validate)
        self.ids = ids
        self.X_train, self.X_test, self.y_train, self.y_test, self.train_ids, self.test_ids = self.split_data(X, y, ids, cross_validate)

    def apply_set(self, bidders):
        # TODO: first apply filtering - then split the data
        self.uX_scaled = []
        self.uy_scaled = []
        n = len(self.ids)
        for i in xrange(n):
            if self.ids[i] in bidders:
                self.uX_scaled.append(self.X_scaled[i])
                self.uy_scaled.append(self.y_scaled[i])


def add_bidder(bidder):
    if bidder.bidder_id in shared.bidders:
        print "duplicate found:",
        print shared.bidders[bidder.bidder_id]
        print bidder
        print

    shared.bidders[bidder.bidder_id] = bidder


def bids_to_features(fit=True):
    X = []
    y = []
    bidder_ids = []
    for bid in shared.bids:
        if bid.outcome is None and fit:
            continue
        if bid.outcome is not None and not fit:
            continue

        bidder_ids.append(bid.bidder_id)
        X.append(bid.features())
        if fit:
            y.append(bid.outcome)

    return X, y, bidder_ids


def bidder_to_features(fit=True):
    X = []
    y = []
    ids = []
    skipped = 0
    skip_hyper = 0
    hc = 0
    rc = 0
    print "Bidder to features fit=%s %s" % (fit, len(shared.bidders))
    for k, v in shared.bidders.iteritems():
        # print len(v.auctions), len(v.bids), v.outcome, len(v.auctions) > 150, v.outcome < 0.1
        if v.outcome is None and fit:
            skipped += 1
            # skip test bidders while doing training
            continue

        if not fit and v.outcome is not None:
            skipped += 1
            # skip train bidders while predicting
            continue

        # if fit and len(v.auctions) > 150 and v.outcome < 0.1:
        #     skip_hyper += 1
        #     continue

        x = v.features()

        # if v.outcome < 0.1 and hc and len(v.auctions_to_bid_idx) < 10:
        #   hc -=1
        #   print "Human: ", v.bidder_id
        #   for auction, b in v.auctions_to_bid_idx.iteritems():
        #       print auction, v.get_avg_update_len_for_auction(auction), b

        # if v.outcome > 0.1 and rc and len(v.auctions_to_bid_idx) < 10:
        #   rc -=1
        #   print "Robot: ", v.bidder_id
        #   for auction, b in v.auctions_to_bid_idx.iteritems():
        #       print auction, v.get_avg_update_len_for_auction(auction), b

        X.append(x)
        if v.outcome is not None:
            y.append(float(v.outcome))
        ids.append(k)

    print "skipped invalid bidders:", skipped
    print "skipped %s hyper active users" % skip_hyper

    return X,y,ids


def print_robot(word, bidder_id, model, i, j, names, probas, conf):
    print "%s (%s)" % (word, conf[i])
    print_X(model.X_unscaled, j)
    for clf, prob in zip(names, probas):
        print '[%s]: %0.3f, %0.3f  ' % (clf[0], prob[i][0], prob[i][1]),
    print
    print "----------------"


def analyze(eclf, model):
    eclf.fit(model.X_train, model.y_train)

    predicted = eclf.predict(model.X_test)
    conf = eclf.predict_proba(model.X_test)
    probas = eclf._predict_probas(model.X_test)
    names = _name_estimators(eclf.clfs)

    # display sample rows of selected features
    print model.selected_feature_names()
    print "Robots"
    # print_bidders(model.X_unscaled, model.y_unscaled, 100, 1.0)
    utils.print_features(model.ids, shared.bidders, model.y_unscaled, 50, 1.0)
    print "Humans"
    # print_bidders(model.X_unscaled, model.y_unscaled, 100, 0.0)
    utils.print_features(model.ids, shared.bidders, model.y_unscaled, 50, 0.0)
    print

    # display found and missed rows with confidence rate
    l = zip(predicted, model.y_test)

    found = 0
    missed = 0
    marked_human_as_robot = 0

    print model.selected_feature_names()
    for i,item in enumerate(l):
        p,k = map(int, item)
        bidder_id = model.test_ids[i]
        j = model.ids.index(bidder_id)  # index of the bidder in not splitted set
        if p == k and k == 1:
            print_robot("found", bidder_id, model, i, j, names, probas, conf)
            found += 1
        elif p != k and k == 0:
            print_robot("wrong label", bidder_id, model, i, j, names, probas, conf)
            marked_human_as_robot += 1
        elif p != k and k == 1:
            print_robot("missed", bidder_id, model, i, j, names, probas, conf)
            missed += 1
    score = eclf.score(model.X_test, model.y_test)
    print 'done analyze: found %s, missed %s, mislabeled %s, score %s' % (found, missed, marked_human_as_robot, score)


def find_wolves():
    print 'Wolves:'
    hw = 0
    rw = 0
    robots = 0
    min_resp_time = sys.maxint
    for k,bidder in shared.bidders.iteritems():
        f = bidder.features()

        if len(bidder.auctions) > 100:
            if utils.is_robot(bidder):
                rw += 1
            elif utils.is_human(bidder):
                hw += 1

            bidder.set_outcome(1.0)


    print 'among %s bidders, found %s wolves-humans and %s robots (among %s robots in general)' % (len(bidders), hw, rw, robots)


def tuning(X, y, clf):
    param_grid = [
        {
            'logisticregression__C': [1.0, 100.0],
            'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__class_weight': ['auto', None],
            'randomforestclassifier__n_estimators': [20, 200, 2000],
            # 'svc__C': [1, 10, 100, 1000],
            # 'svc__kernel': ['linear'],
            # 'svc__degree': [1,2,3],
            # 'svc__gamma': [0.001, 0.0001]
        }
    ]
    # param_grid = [
    #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    #     {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    # ]

    param_grid = [
        {
            'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
            'adaboostclassifier__n_estimators': [20, 200, 2000],
            'adaboostclassifier__learning_rate': [0.01, 0.1, 1],
        }
    ]
    """
    param_grid = [
            {
                'gradientboostingclassifier__learning_rate': [1, 0.1, 0.01],
                'gradientboostingclassifier__subsample': [1, 0.5, 0.1],
                'gradientboostingclassifier__max_features': [None, 'log2', 2, 5]
            }
    ]
    """
    param_grid = [
            {
                'baggingclassifier__n_estimators': [10, 100, 200],
                'baggingclassifier__max_samples': [1.0, 0.7, 0.5, 0.1],
                'baggingclassifier__max_features': [1.0, 0.7, 0.5],
                'baggingclassifier__bootstrap': [True, False],
                'baggingclassifier__bootstrap_features': [True, False]
            }
    ]
    scores = ['roc_auc'] #['precision', 'recall']
    for score in scores:
        print "# Tuning hyper-parameters for %s" % score
        print

        grid_srch = GridSearchCV(clf, param_grid, cv=5, scoring=score)# scoring='%s_weighted' % score)
        grid_srch.fit(X, y)

        print "Best parameters set found on development set:"
        print
        print grid_srch.best_params_
        print
        print "Grid scores on development set:"
        print
        for params, mean_score, scores in grid_srch.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print


def cv_fit(X, y, fit, validate, clfs, labels, tune=False):
    eclf = clfs[-1]
    if validate:
        if tune:
            tuning(X, y, eclf)
            print "Done tuning"

        print "5-fold cross validation:"

        for clf, label in zip(clfs, labels):
            scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            print "ROC Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)

            scores = cross_validation.cross_val_score(clf, X, y, cv=5)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            # pred_y = clf.predict_proba(model.X_test)[:, 1]
            # print "ROC Score: %0.2f [%s]" % (roc_auc_score(np.array(model.y_test).astype(float), np.array(pred_y).astype(float)), label)
            # if label != 'EnsembleClassifier':
            # EnsembleClassifier graphs don't work due to error
            utils.draw_learning_curve(X, y, clf, label)
            #utils.draw_ROC(X, y, clf, label)
            print "-----"

        # analyze(eclf, model)
    elif fit:
        print "Fit model with %s samples" % len(X)
        eclf.fit(X, y)

def bids_work(fit, validate, tune=False):
    print "Fit/predict bids"
    model = Model()
    X,y,bidder_ids = bids_to_features(fit)
    model.standard_prepare(X,y,fit,validate)

    clf1 = linear_model.LogisticRegression(penalty='l2', C=100.0)
    clf2 = RandomForestClassifier(n_estimators=20)
    clf3 = svm.SVC(probability=True, kernel='rbf', C=1, gamma = 0.0001)
    clf4 = GradientBoostingClassifier(loss='exponential')
    clf5 = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
    clf6 = KNeighborsClassifier()
    clf7 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME.R",
                         n_estimators=2000,
                         learning_rate=0.01)

    eclf = EnsembleClassifier(clfs=[clf1, clf2], voting='soft')

    cv_fit(model.X_scaled, model.y_scaled, fit, validate, [clf1, clf2, clf3, clf4, clf7, eclf],
        ['LogisticRegression', 'RandomForest', 'SVC', 'GradientBoosting', 'AdaBoost', 'EnsembleClassifier'], tune)

    return model, eclf


def work(fit, validate,tune=False):
    print "Fit/predict bidders"

    model = Model()
    X,y,ids = bidder_to_features(fit)
    model.prepare(X,y,ids,fit,validate)

    clf1 = linear_model.LogisticRegression(penalty='l2', C=100.0)
    clf2 = RandomForestClassifier(n_estimators=20)
    clf3 = svm.SVC(probability=True, kernel='rbf', C=1, gamma = 0.0001)
    clf4 = GradientBoostingClassifier(loss='exponential', subsample=1, max_features='log2', learning_rate=0.1)
    clf5 = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
    clf6 = BaggingClassifier(n_jobs=7)
    clf7 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME.R",
                         n_estimators=2000,
                         learning_rate=0.01)

    eclf = EnsembleClassifier(clfs=[clf1, clf2], voting='soft')

    # bidder_sets = split_bidders()
    # for bs in bidder_sets:
    #     model.apply_set(bs)
    #     cv_fit(model.uX_scaled, model.uy_scaled, fit, validate, clf1, clf2, clf3, clf4, eclf)

    cv_fit(model.X_scaled, model.y_scaled, fit, validate, [clf1, clf2, clf3, clf4, clf7, eclf],
         ['LogisticRegression', 'RandomForest', 'SVC', 'GradientBoosting', 'AdaBoost', 'EnsembleClassifier'], tune)

    #cv_fit(model.X_scaled, model.y_scaled, fit, validate, [eclf],
    #    ['BaggingClassifier'], tune)

    return model, eclf


def predict(model, clf):
    X,y,ids = bidder_to_features(False)
    print "Prepare data"
    model.prepare(X,y,ids,False,False)
    print "Start prediction"
    predicted = clf.predict(model.X_scaled)
    conf = clf.predict_proba(model.X_scaled)
    print "Write results"
    with open('result.csv', 'wb') as result:
        fieldnames = ['bidder_id','prediction']
        writer = csv.DictWriter(result, fieldnames=fieldnames)
        writer.writeheader()
        robots = 0
        humans = 0
        for i in xrange(len(X)):
            writer.writerow({ "bidder_id": ids[i], "prediction":  "%1.13f" % conf[i][1] })
            if conf[i][1] > 0.5:
                robots += 1
            else:
                humans += 1

        print "predicted %s robots and %s humans" % (robots, humans)


def bids_predict(model, clf):
    X,y,bidder_ids = bids_to_features(False)
    print "Prepare data"
    model.standard_prepare(X,y,False,False)

    print "Start prediction"
    predicted = clf.predict(model.X_scaled)
    conf = clf.predict_proba(model.X_scaled)
    print "Define bidders guilt"
    n = len(X)
    sp = defaultdict(list)
    for bp, bidder_id in zip(conf, bidder_ids):
        p = bp[1]
        sp[bidder_id].append(p)

    print "Write results"
    with open('result.csv', 'wb') as result:
        fieldnames = ['bidder_id','prediction']
        writer = csv.DictWriter(result, fieldnames=fieldnames)
        writer.writeheader()
        robots = 0
        humans = 0
        for bidder_id, probs in sp.iteritems():
            avg_p = np.average(probs)
            writer.writerow({ "bidder_id": bidder_id, "prediction":  "%1.13f" % avg_p })
            if avg_p > 0.5:
                robots += 1
            else:
                humans += 1

        print "predicted %s robots and %s humans" % (robots, humans)


def get_median_price_per_product(price_per_auction):
    median_price_per_product = defaultdict(int)
    std = defaultdict(float)
    price_per_product = defaultdict(list)
    for auc, price in price_per_auction.iteritems():
            prod = shared.auctions_to_products[auc]
            price_per_product[prod].append(price)

    for prod, prices in price_per_product.iteritems():
        median_price_per_product[prod] = np.median(prices)
        std[prod] = np.std(prices)

    return median_price_per_product, std


def increments_per_bidder_per_auction():
    for auc, increments in shared.auction_to_increments.iteritems():
        c = 0
        for inc in increments:
            if inc.bidder_id in shared.bidders:
                shared.bidders[inc.bidder_id].increments_per_auction[inc.auction].append(inc)
            if inc.is_robot:
                c += 1
            if inc.is_human:
                c += 1


def calc_auction_rank():
    print "Determine auction rank"
    for auc, bids in shared.auction_to_bids.iteritems():
        for bid in bids:
            if bid.is_robot:
                shared.auction_rank[auc] += 1
        if bids:
            shared.auction_rank[auc] /= float(len(bids))



def load_increments():
    with open('increments.csv', 'rb') as incfile:
        reader = csv.reader(incfile, delimiter=',', quotechar='|')
        skip(reader, 1)
        for row in reader:
            inc = StoredIncrement(*row)
            shared.auction_to_increments[inc.auction].append(inc)


def save_increments():
    for auc, bids in shared.auction_to_bids.iteritems():
        prev_bid = None
        inc = Increment()
        for bid in bids:
            if prev_bid is None or prev_bid.bidder_id == bid.bidder_id:
                inc.bids.append(bid)
            elif prev_bid is not None and prev_bid.bidder_id != bid.bidder_id:
                shared.auction_to_increments[auc].append(inc)
                inc = Increment()
                inc.bids.append(bid)

            prev_bid = bid
        # last increment
        if inc.bids:
            shared.auction_to_increments[prev_bid.auction].append(inc)

    with open('increments.csv', 'wb') as result:
        fieldnames = ['auction', 'bidder_id', 'time', 'price', 'diff_price', 'is_human', 'is_robot', 'is_last', 'merchandise', 'ips_count', 'country']
        writer = csv.DictWriter(result, fieldnames=fieldnames)
        writer.writeheader()
        for auc, increments in shared.auction_to_increments.iteritems():
            for inc in increments:
                row = {
                    'auction': auc,
                    'bidder_id': inc.bidder_id,
                    'time': inc.time,
                    'price': inc.price,
                    'diff_price': inc.diff_price,
                    'is_human': inc.is_human,
                    'is_robot': inc.is_robot,
                    'is_last': inc.is_last,
                    'merchandise': inc.merchandise,
                    'ips_count': inc.ips_count,
                    'country': inc.country
                }
                writer.writerow(row)


def response_time_per_bidder_per_auction():
    for auc, bids in shared.auction_to_bids.iteritems():
        prev_bid = None
        for bid in bids:
            if prev_bid and prev_bid.bidder_id != bid.bidder_id:
                shared.bidders[bid.bidder_id].resp_times_auction.append(bid.time-prev_bid.time)
            prev_bid = bid


def response_time_per_bidder():
    """
    Response time of the bidder in different auctions.
    For example user made a bid in auction A and after 1s in auction B.
    This 1s should be recorded. It doesn't record consequentive bids in the same auction.
    """
    last_bidder_bid = dict()
    for bid in shared.bids:
        bidder_id = bid.bidder_id
        if bidder_id in last_bidder_bid:
            last_bid = last_bidder_bid[bidder_id]
            if last_bid.auction != bid.auction:
                shared.bidders[bidder_id].resp_times.append(bid.time-last_bid.time)
        last_bidder_bid[bidder_id] = bid


def search_group_patterns(auction_to_increments):
    # Find longest subsequence of elements with fixed delta d
    for auc, increments in auction_to_increments.iteritems():
        T = [i.time for i in increments if i.is_robot]
        n = len(T)
        if not n:
            continue
        # L[i][j] - length of sequence with delta j ending on element T[i]
        D = max(T) - min(T)
        L = [[0 for j in xrange(D+1)] for i in xrange(n)]
        for i in xrange(1, n):
            for j in xrange(i):
                d = T[i]-T[j]
                L[i][d] = max(L[i][d], L[j][d]+1, 2)

        l = 0
        ld = 0
        for i in xrange(n):
            for d in xrange(D+1):
                if L[i][d] > l:
                    l = L[i][d]
                    ld = d
        if l > 2:
            print l, ld


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def has_periods(auction_to_increments):
    allowed = ['ca82l', 'qi6xz', 'sn8b3', 'q925k', '948q5', '2hu2f', 'mbkj7']
    for auc, increments in auction_to_increments.iteritems():
        if auc not in allowed:
            continue
        _min = increments[0].time
        _max = increments[-1].time
        n = _max-_min+1
        four_hours = 60*60*4
        if n < four_hours:
            return []

        T = [0 for i in xrange(n)]
        for inc in increments:
            T[inc.time-_min] = 1

        utils.plot_moving_avg(auc, moving_average(T, four_hours))


class TreeNode():
    def __init__(self, id):
        self.id = id
        self.leader = None
        self.children = list()

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return "%s (%s)" % (str(self.id), len(self.children))

    def __eq__(self, other):
        return self.id == other.id


def merge_trees(u, v):
    if u.leader:
        u = u.leader
    if v.leader:
        v = v.leader

    if v == u:  # already merged
        return

    if len(u.children) < len(v.children):
        u, v = v, u
    # u > v from now on
    v.leader = u
    u.children.append(v)
    for c in v.children:
        c.leader = u
        u.children.append(c)


def split_bidders():
    print "Split bidders"
    nodes = dict()
    for auc, bids in shared.auction_to_bids.iteritems():
        leader = None
        for bid in bids:
            if bid.bidder_id not in nodes:
                nodes[bid.bidder_id] = TreeNode(bid.bidder_id)
            node = nodes[bid.bidder_id]
            if not leader:
                # choose leader for this auction
                if node.leader:
                    leader = node.leader
                else:
                    leader = node
            else:
                merge_trees(leader, node)

    leaders = [u for u in nodes.values() if u.leader is None]
    sets = []
    for leader in leaders:
        s = [leader.id] + [c.id for c in leader.children]
        if len(s) > 1:
            sets.append(s)

    return sets



def build_graph():
    G = nx.Graph()
    G.add_nodes_from(shared.bidders.keys())
    user_to_auc = defaultdict(set)
    # for auc, bids in shared.auction_to_bids.iteritems():
    #     users = list(set([bid.bidder_id for bid in bids]))
    #     n = len(users)
    #     for i in xrange(n):
    #         for j in xrange(n):
    #             G.add_edge(users[i], users[j])

    for auc, bids in shared.auction_to_bids.iteritems():
        for bid in bids:
            user_to_auc[bid.bidder_id].add(auc)

    pair_to_user = defaultdict(set)
    for bidder_id, aucs in user_to_auc.iteritems():
        n = len(aucs)
        a = list(aucs)
        for i in xrange(n):
            for j in xrange(i+1, n):
                u = min(a[i], a[j])
                v = max(a[i], a[j])
                pair_to_user[(u,v)].add(bidder_id)

    for pair, bidders in pair_to_user:
        n = len(bidders)
        b = list(bidders)
        for i in xrange(n):
            for j in xrange(i+1,n):
                G.add_edge(bidders[i], bidders[j])

    utils.draw_graph(G, shared.bidders)


def save_bidders_sets():
    bidder_sets = split_bidders()
    for i in xrange(len(bidder_sets)):
        s = bidder_sets[i]
        for j in ['train', 'test']:
            with open('set_%s_%s.csv' % (i, j), 'wb') as setfile:
                fieldnames = ['bidder_id', 'payment_account', 'address']
                if j == 'train':
                    fieldnames += ['outcome']
                writer = csv.DictWriter(setfile, fieldnames=fieldnames)
                writer.writeheader()
                for bidder_id in s:
                    b = shared.bidders[bidder_id]
                    if (j == 'train' and b.outcome is None) or (j == 'test' and b.outcome is not None):
                        continue

                    row = {
                        'bidder_id': b.bidder_id,
                        'payment_account': b.payment_account,
                        'address': b.address
                    }
                    if j == 'train':
                        row['outcome'] = b.outcome
                    writer.writerow(row)


def co_occurrence():
    print "Count bidders co-occurrence"
    for auc, increments in shared.auction_to_bids.iteritems():
        users = list(set([inc.bidder_id for inc in increments if (inc.is_human or inc.is_robot)]))
        n = len(users)
        for i in xrange(n):
            for j in xrange(i+1,n):
                u = min(users[i], users[j])
                v = max(users[i], users[j])
                shared.pairs[(u,v)] += 1
    print "Total pairs: %s" % (len(shared.pairs))
    P = [(c, pair) for pair, c in shared.pairs.iteritems()]
    P.sort(reverse=True)
    print "Top co-occurences:"
    for i in xrange(200):
        count, pair = P[i]
        u, v = pair
        print "%s (%s) and %s (%s) seen together %s times" % (u, utils.get_bidder_label(u), v, utils.get_bidder_label(v), count)
        print "%s user participated in %s auctions" % (u, len(shared.bidders[u].auctions))
        print "%s user participated in %s auctions" % (v, len(shared.bidders[v].auctions))

    for auc, increments in shared.auction_to_bids.iteritems():
        prev = None
        for inc in increments:
            if prev is None:
                prev = inc
            else:
                shared.pairs[(prev.bidder_id, inc.bidder_id)] += 1
                prev = inc

    print "Top 'bid-after' for pairs of users:"
    P = [(c, pair) for pair, c in shared.pairs.iteritems()]
    P.sort(reverse=True)
    for i in xrange(200):
        count, pair = P[i]
        u, v = pair
        print "user %s (%s) placed a bid right after %s (%s) - %s times" % (u, utils.get_bidder_label(u), v, utils.get_bidder_label(v), count)
        print "%s user placed %s bids" % (u, len(shared.bidders[u].bids))
        print "%s user placed %s bids" % (v, len(shared.bidders[v].bids))


def analyze_time():
    # min, max, min diff in a day, max diff in a day, last seen time
    _min = sys.maxint
    _max = 0
    diff_min = sys.maxint
    diff_max = 0
    diffs = []
    for auc, bids in shared.auction_to_bids.iteritems():
        prev_bid = None
        for bid in bids:
            _min = min(_min, bid.time)
            _max = max(_max, bid.time)

            if prev_bid:
                diff_min = min(diff_min, bid.time - prev_bid.time)
                diff_max = max(diff_max, bid.time - prev_bid.time)
                diffs.append(bid.time - prev_bid.time)

            prev_bid = bid

    print "Day min: %s, max: %s, min diff.: %s, max diff.: %s" % (_min, _max, diff_min, diff_max)
    print "Diffs median: %.2f, average: %.2f, std: %.2f" % (np.median(diffs), np.average(diffs), np.std(diffs))

hash_masks = dict()

def get_hash(i):
    mask = hash_masks.get(i)
    if mask is None:
        random.seed(i)
    mask = hash_masks[i] = random.getrandbits(32)

    def myhash(x):
        return hash(x) ^ mask

    return myhash

def get_hashes(n):
    hashes = []
    for i in xrange(n):
        hashes.append(get_hash(i))
    return hashes

def get_sim_rate(S, i, j):
    a = S[:, i]
    b = S[:, j]
    u = 0
    for k in xrange(len(a)):
        if a[k] == b[k]:
            u += 1

    return float(u)/len(a)

def sim_bidders():
    """
    Calculates similarity between users bids tracks (track - bidding strategy within auction)
    """
    tracks = defaultdict(dict)
    for bid in shared.bids:
        tracks['%s_%s' % (bid.bidder_id, bid.auction)][bid.time] = True

    def print_result(a, b, rate):
        la = utils.get_bidder_label(a.split('_')[0])
        lb = utils.get_bidder_label(b.split('_')[0])
        print "Tracks %s (%s) and %s (%s) are similar %.2f" % (a, la, b, lb, rate)


    lsh(tracks, 'tracks_signature.pkl', 100, print_result)

def sim_auctions():
    """
    Calculates similarity between auctions
    """
    tracks = defaultdict(dict)
    for bid in shared.bids:
        tracks[bid.auction][bid.time] = True

    def print_result(a, b, rate):
        print "Auctions %s and %s are similar %.2f" % (a, b, rate)

    lsh(tracks, 'auctions_signature.pkl', 100, print_result)

def lsh(tracks, filename, hn, print_result):
    n = len(tracks)
    ids = tracks.keys()

    hashes = get_hashes(hn)

    tmax = 260000
    S = [[sys.maxint for c in xrange(n)] for i in xrange(hn)]  # signature matrix

    for i in xrange(tmax):  # loop over time
        for j in xrange(hn):  # loop over hash functions
            v = hashes[j](i) % tmax
            for c in xrange(n):  # loop over users
                track_id = ids[c]
                if i in tracks[track_id]:
                    S[j][c] = min(S[j][c], v)

    joblib.dump(S, filename, compress = 3)

    print "Split into buckets"
    B = 10  # bucket size
    R = 10  # number of buckets
    T = math.pow((1/b), (1/r))  # approx. 0.89 threshold for similarity
    buckets = defaultdict(list)
    for r in xrange(R):
        for j in xrange(n):
            h = hash(tuple(S[r*B:(r+1)*B,j]))
            buckets[h].append(j)

    print "Find similar"
    for k, candidates in buckets.iteritems():
        m = len(candidates)
        for i in xrange(m):
            for j in xrange(i+1, m):
                rate = get_sim_rate(S, i, j)
                if rate >= T:
                    a = ids[i]
                    b = ids[j]
                    print_result(a, b, rate)


def read_bids(filename):
    country_to_time = read_country_to_time()
    skip_bids_count = 0
    skip_aucs = set()
    # time_user = dict()
    shared.price_threshold = 2000

    with open(filename, 'rb') as bidsfile:
        reader = csv.reader(bidsfile, delimiter=',', quotechar='|')
        skip(reader, 1)
        c = 0
        prev_bid = None

        days = [
            [sys.maxint, 0, sys.maxint, 0, 0],  # min, max, min diff in a day, max diff in a day, last seen time
            [sys.maxint, 0, sys.maxint, 0, 0],
            [sys.maxint, 0, sys.maxint, 0, 0]
        ]

        for row in reader:
            c += 1
            if c % 1e6 == 0:
                print "Read bids progress: %s" % c

            bidder_id = row[1]

            if bidder_id not in shared.bidders:# or (shared.bidders[bidder_id].outcome is None and fit) or (shared.bidders[bidder_id].outcome is not None and not fit):
                continue

            bid = Bid(*row)

            # if bid.time not in time_user:
            #     time_user[bid.time] = defaultdict(list)
            # time_user[bid.time][bid.bidder_id].append(bid)

            shared.bids.append(bid)

            # time_prob = 0

            # if bid.country in country_to_time:
            #     time_prob = country_to_time[bid.country][time_to_hour(bid.time)]

            shared.auctions_to_products[bid.auction] = bid.merchandise
            shared.products.add(bid.merchandise)
            shared.countries.add(bid.country)

            shared.bidders[bidder_id].update(bid, 0)  # Ahutng!

            shared.auction_to_bids[bid.auction].append(bid)

            if bid.is_human or bid.is_robot:
                # gather stats on ip
                pref_ip = bid.ip_pref
                if pref_ip not in shared.ip_to_bidder:
                    shared.ip_to_bidder[pref_ip] = [0, 0]
                shared.ip_to_bidder[pref_ip][0 if bid.is_human else 1] += 1

            prev_bid = bid

    print "Sim bidders"
    sim_bidders()

    print "Bids stats"
    print "Total bids: %s, total auctions: %s" % (len(shared.bids), len(shared.auction_to_bids))

    for auc, bids in shared.auction_to_bids.iteritems():
        if not bids:
            continue
        all_unique = set()
        c = defaultdict(int)
        q = deque()
        u = 0
        first_bid = bids[0]
        last_bid = bids[-1]
        prev = None
        l = float(last_bid.time - first_bid.time)
        for bid in bids:
            if bid.bidder_id not in c:
                u += 1

            c[bid.bidder_id] += 1
            q.append(bid.bidder_id)
            all_unique.add(bid.bidder_id)

            if len(q) > 50:
                rid = q.popleft()
                c[rid] -= 1
                if not c[rid]:
                    u -= 1

            bid.prev_unique = len(all_unique)
            bid.prev_unique_50 = u
            bid.time_from_start = bid.time - first_bid.time
            bid.time_to_end = last_bid.time - bid.time
            bid.auc_length = l

            if prev:
                bid.time_to_prev_bid = bid.time - prev.time
            prev = bid

    print "Time stats"

    h = []
    r = []
    bins = [i for i in xrange(0, 265000, 5000)]
    for auc, bids in shared.auction_to_bids.iteritems():
        for bid in bids:
            j = bisect.bisect_left(bins, bid.time)
            if j >= len(bins):
                print j, bid.time
            if bid.is_human:
                h.append(bins[j])
            elif bid.is_robot:
                r.append(bins[j])

    shared.human_hist, shared.human_hist_bins = np.histogram(h, bins, density=True)
    joblib.dump(shared.human_hist, 'human_hist.pkl', compress = 3)
    joblib.dump(shared.human_hist_bins, 'human_hist_bins.pkl', compress = 3)

    #shared.human_hist = joblib.load('human_hist.pkl')
    #shared.human_hist_bins = joblib.load('human_hist_bins.pkl')
    print "Human hist prob"
    print shared.human_hist
    print shared.human_hist_bins

    analyze_time()

    print "Determine increments"
    save_increments()
    #print "Load increments"
    #load_increments()
    increments_per_bidder_per_auction()
    calc_auction_rank()

    shared.countries = list(shared.countries)
    shared.countries.sort()

    #print "Search group patterns in increments"
    #search_group_patterns(shared.auction_to_increments)
    #print "Search group patterns in bids"
    #search_group_patterns(shared.auction_to_bids)
    #print "done search group patterns"
    # save_bidders_sets()
    # sys.exit()

    print "Determine country rank"
    n = len(shared.countries)
    rc = [0 for i in xrange(n)]
    hc = [0 for i in xrange(n)]
    for auc, increments in shared.auction_to_bids.iteritems():
        for inc in increments:
            if inc.is_robot:
                rc[shared.countries.index(inc.country)] += 1
            elif inc.is_human:
                hc[shared.countries.index(inc.country)] += 1

    for i in xrange(n):
        c = shared.countries[i]
        if (rc[i] + hc[i]):
            shared.country_rank[c] = float(rc[i])/(rc[i] + hc[i])

    print "Calculate median price per product"

    for auc, bids in shared.auction_to_bids.iteritems():
        human_prices = [bid.price for bid in bids if bid.is_human and bid.is_last]
        robot_prices = [bid.price for bid in bids if bid.is_robot and bid.is_last]

        shared.human_median_price_per_auction[auc] = np.median(human_prices) if human_prices else 0
        shared.robot_median_price_per_auction[auc] = np.median(robot_prices) if robot_prices else 0

    shared.human_median_price_per_product, shared.human_std_price_per_product = get_median_price_per_product(shared.human_median_price_per_auction)
    shared.robot_median_price_per_product, shared.robot_std_price_per_product = get_median_price_per_product(shared.robot_median_price_per_auction)

    print "Draw graphs"
    utils.plot_median_price_per_product(shared.human_median_price_per_product,
        shared.human_std_price_per_product,
        shared.robot_median_price_per_product,
        shared.robot_std_price_per_product)

    utils.draw_auctions_bids(shared.auction_to_bids, 100)
    utils.draw_bids_time_hist(19011)
    utils.draw_bids_unique(19011)
    # for auc in ['2dfh7_2', 'nys0k_2', 'oqlkh_1', 'qj8uk_2']:
    #     utils.draw_auction(auc)
    #sys.exit()
    """
    utils.draw_robots_country()
    utils.plot_price_per_product_per_bucket(shared.auction_to_increments, shared.price_threshold)

    utils.draw_bidder_to_increments_time(shared.auction_to_increments, 'is_human', 'human_to_increment_time.png')
    utils.draw_bidder_to_increments_time(shared.auction_to_increments, 'is_robot', 'robot_to_increment_time.png')


    """
    print "Total auctions %s" % (len(shared.auction_to_bids))


def read_country_to_time():
        ctt = dict()
        with open('country_to_time.csv', 'rb') as cfile:
            reader = csv.reader(cfile,  delimiter=',',  quotechar='|')
            skip(reader, 1)
            for row in reader:
                ctt[row[0]] = map(float, row[1:])

        return ctt


def read_set0(filename):
    with open(filename, 'rb') as bidsfile:
        reader = csv.reader(bidsfile,  delimiter=',',  quotechar='|')
        skip(reader, 1)
        for row in reader:
            shared.set0.add(row[0])


def read_bidders(filename):
    with open(filename, 'rb') as bidsfile:
        reader = csv.reader(bidsfile,  delimiter=',',  quotechar='|')
        skip(reader, 1)
        for row in reader:
            b = Bidder(row[0], row[1], row[2], row[3] if len(row) > 3 else None)
            #if b.bidder_id in ['f5b2bbad20d1d7ded3ed960393bec0f40u6hn', 'e90e4701234b13b7a233a86967436806wqqw4']:
            #    b.outcome = 1

            add_bidder(b)

            shared.addr_1[b.get_addr_1()].add(b.bidder_id)
            shared.addr_2[b.get_addr_2()].add(b.bidder_id)

            shared.pmt_type[b.get_payment_type()].add(b.bidder_id)
            shared.pmt_accnt[b.get_payment_acct()].add(b.bidder_id)

        print "%s data done" % filename
        rc = sum([1 for i in shared.bidders.values() if utils.is_robot(i)])
        hc = sum([1 for i in shared.bidders.values() if utils.is_human(i)])
        uc = sum([1 for i in shared.bidders.values() if i.outcome is None])
        print "Found %s robots, %s humans, %s unknown" % (rc, hc, uc)


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest="train", action='store_true')
parser.add_argument('--tune', dest="tune", action='store_true')
args = parser.parse_args()

read_bidders('train.csv')
read_bidders('test.csv')
read_bids('clean_bids.csv')

read_set0('set_0_train.csv')
read_set0('set_0_test.csv')

if args.train:
    # model, clf = bids_work(fit=True,validate=True,tune=args.tune)
    model, clf = work(fit=True,validate=True,tune=args.tune)
else:
    # model, clf = bids_work(fit=True,validate=False)
    # bids_predict(model, clf)
    model, clf = work(fit=True,validate=False)
    predict(model, clf)
