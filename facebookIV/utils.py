import numpy as np
import networkx as nx
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import roc_curve, auc
from sklearn.learning_curve import learning_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split

from datetime import datetime
from collections import defaultdict, deque
import shared

class RequiredSelector(VarianceThreshold):
    def __init__(self, *args, **kwargs):
        self.required = kwargs['required'] if 'required' in kwargs else None
        super(RequiredSelector, self).__init__(*args, **kwargs)

    def set_required(self, required):
        self.required = required

    def fit(self, X, y=None):
        super(RequiredSelector, self).fit(X,y)
        for i in self.required:
            self.variances_[i] = self.threshold + 0.001


class StupidClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.array([int(p[1]) for p in prob])

    def predict_proba(self, X):
        prob = []
        for x in X:
            if (x > 0).all():
                prob.append(np.array([0.0, 1.0]))
            else:
                prob.append(np.array([0.0, 0.0]))

        return prob


def is_robot(bidder):
    return bidder.outcome is not None and bidder.outcome > 0.1


def is_human(bidder):
    return bidder.outcome is not None and bidder.outcome < 0.1


def is_unknown(bidder):
    return bidder.outcome is not None


def get_bidder_label(bidder_id):
    bidder = shared.bidders[bidder_id]
    if is_robot(bidder):
        return 'robot'
    if is_human(bidder):
        return 'human'
    return 'unknown'

def skip(file, n):
    for _ in xrange(n):
        next(file)


def print_X(X, i):
    for x in X[i]:
        print "%.2f" % x,
    print


def print_bidders(X, y, c, isRobot):
    for i in xrange(len(X)):
        if abs(y[i] - isRobot) < 0.1:
            print_X(X, i)
            c -= 1
            if not c:
                break


def print_features(ids, bidders, y, c, isRobot):
    for i in xrange(len(ids)):
        if abs(y[i] - isRobot) < 0.1:
            print bidders[ids[i]].features()
            c -= 1
            if not c:
                break


m24 = 24 * 60 * 60 * 1000


def time_to_hour(time):
    global m24
    ms = (int(time)-9631916842105263) % m24 + 0.0
    return int(round(ms / (1000*60*60))) % 24


def divide(a, b):
    if b == 0:
        return -1
    r = float(a)/b

    if not r:
        return 0.00000000000000000000001

    return r


def short_float(a):
    return round(a, 2)


def get_ip_part(ip, parts):
    ip = ip.split('.')
    return '.'.join(ip[:parts])

def get_ip_pref(ip):
    return get_ip_part(ip, 2)


def get_price_bucket_name(price_buckets, i):
    if i == 0 or i == len(price_buckets):
        print "bad i", i
        return "bad_i"
    return 'price_' + str(price_buckets[i-1]) + '_' + str(price_buckets[i])


def draw_ROC(X, y, clf, title):
    rcParams['figure.figsize'] = 16, 12
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    clf.fit(X_train, y_train)

    # Determine the false positive and true positive rates
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    # print '[%s] ROC AUC: %0.2f' % (title, roc_auc)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s ROC Curve' % title)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("roc_%s.png" % title, bbox_inches='tight')
    plt.close()


def draw_learning_curve(X, y, clf, title):
    rcParams['figure.figsize'] = 16, 12
    ## Adapted from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
    # assume classifier and training data is prepared...

    train_sizes, train_scores, test_scores = learning_curve(
            clf, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim((0.6, 1.01))
    plt.gca().invert_yaxis()
    plt.grid()

    # Plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
    plt.legend(loc="best")

    # Plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")

    # Draw the plot and reset the y-axis
    plt.draw()
    # plt.show()
    plt.savefig("learning_curve_%s.png" % title, bbox_inches='tight')
    plt.gca().invert_yaxis()
    plt.close()


def plot_median_price_per_product(human_median_price_per_product, human_std_price_per_product, robot_median_price_per_product, robot_std_price_per_product):
    rcParams['figure.figsize'] = 16, 12
    # TODO: plot price per product divided into "cheap" and "expensive" categories each
    # TODO: plot human/robot last price per product
    # TODO: plot human/robot won price per product
    width = 0.5
    bins = []
    fig, ax = plt.subplots()
    hp = []
    hp_std = []
    rp = []
    rp_std = []
    for prod, pr in human_median_price_per_product.iteritems():
        bins.append(prod)
        hp.append(pr)
        hp_std.append(human_std_price_per_product[prod])

        rp.append(robot_median_price_per_product[prod])
        rp_std.append(robot_std_price_per_product[prod])

    ind = np.arange(len(bins))
    rects1 = ax.bar(ind, hp, width, color='g', alpha=0.7, yerr=hp_std)
    rects2 = ax.bar(ind+width, rp, width, color='r', alpha=0.5, yerr=rp_std)
    ax.set_ylabel("Price")
    ax.set_xticks(ind+width)
    ax.set_xticklabels(bins)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    plt.savefig("prices.png",bbox_inches='tight')
    plt.close()


def plot_price_bucket(products, humans, robots, j, pos, title):
    ax = plt.subplot(pos)
    width = 0.5
    n = len(products)
    ind = np.arange(n)
    hp = [0 for i in xrange(n)]
    rp = [0 for i in xrange(n)]
    hp_std = [0 for i in xrange(n)]
    rp_std = [0 for i in xrange(n)]
    for i in xrange(n):
        hp[i] = np.average(humans[j][products[i]]) if humans[j][products[i]] else 0
        hp_std[i] = np.std(humans[j][products[i]]) if humans[j][products[i]] else 0
        rp[i] = np.average(robots[j][products[i]]) if robots[j][products[i]] else 0
        rp_std[i] = np.std(robots[j][products[i]]) if robots[j][products[i]] else 0

    rects1 = ax.bar(ind, hp, width, color='g', alpha=0.7, yerr=hp_std)
    rects2 = ax.bar(ind+width, rp, width, color='r', alpha=0.5, yerr=rp_std)
    ax.set_ylabel("Price")
    ax.set_xlabel(title)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(products)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')


def plot_price_per_product_per_bucket(auction_to_increments, price_th):
    rcParams['figure.figsize'] = 16, 24
    humans = [defaultdict(list), defaultdict(list)]
    robots = [defaultdict(list), defaultdict(list)]
    products = set()

    for auc, increments in auction_to_increments.iteritems():
        for inc in increments:
            i = 0
            if inc.price > price_th:
                i = 1

            if inc.is_human:
                humans[i][inc.merchandise].append(inc.price)
            elif inc.is_robot:
                robots[i][inc.merchandise].append(inc.price)

            products.add(inc.merchandise)

    products = list(products)

    plt.figure(1)
    plot_price_bucket(products, humans, robots, 0, 211, "Cheaper than $%s" % price_th)
    plot_price_bucket(products, humans, robots, 1, 212, "Expensive than $%s" % price_th)

    plt.savefig("prices_per_bucket.png",bbox_inches='tight')
    plt.close()


def draw_bidder_to_increments_time(auctions_to_bids, attr, filename):
    rcParams['figure.figsize'] = 256, 128
    robots = defaultdict(list)
    for auc, bids in auctions_to_bids.iteritems():
        for bid in bids:
            if getattr(bid, attr):
                robots[bid.bidder_id].append(bid.time)
    # transform dictionary to 2D plot
    X = []
    Y = []
    ids = robots.keys()
    for i in xrange(len(ids)):
        for y in robots[ids[i]]:
            X.append(i)
            Y.append(y)

    fig, ax = plt.subplots()
    offsets = [i*2 for i in xrange(len(robots))]

    def format_date(y, pos=None):
        return datetime.fromtimestamp(y).strftime("%b %d %H:%M:%S")

    ax.yaxis.grid(True)
    ax.yaxis.set_major_locator(MaxNLocator(64))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()

    ax.set_xticks(offsets)
    ax.set_xticklabels(ids)
    # plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    c = [1, 0, 0]
    if attr == 'is_human':
        c = [0, 0, 1]
    plt.scatter(X, Y, c=c, s=20, marker='_')

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def get_color(bid):
    if bid.is_human:
        return [0,0,1]
    elif bid.is_robot:
        return [1,0,0]
    return [0,1,0]


def draw_auction(auc):
    rcParams['figure.figsize'] = 48, 16
    bids = shared.auction_to_bids[auc]

    bidders = list(set([bid.bidder_id for bid in bids]))
    print "Draw auction %s with %s bidders" % (auc, len(bidders))

    X = []
    Y = []
    colors = []

    i = 0
    for bidder_id in bidders:
        for bid in bids:
            if bid.bidder_id != bidder_id:
                continue

            X.append(i)
            Y.append(bid.time)
            colors.append(get_color(bid))
        i += 1


    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    ax.yaxis.set_major_locator(MaxNLocator(64))
    ax.xaxis.grid(True)
    plt.xlabel(shared.auctions_to_products[auc])

    ax.set_xticks([i for i in xrange(len(bidders))])
    ax.set_xticklabels(bidders)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')

    plt.scatter(X, Y, c=colors, s=20, marker='_')
    plt.savefig("auctions/auction_%s.png" % auc, bbox_inches='tight')
    plt.close()


def get_random_auctions(n):
    keys = shared.auction_to_bids.keys()
    idx = [i for i in xrange(len(keys))]
    np.random.shuffle(idx)
    result = []
    for i in xrange(n):
        result.append(keys[idx[i]])
    return result


def draw_bids_time_hist(n):
    aucs = get_random_auctions(n)
    h = []
    r = []
    bins = [i for i in xrange(0, 265000, 5000)]
    for auc in aucs:
        for bid in shared.auction_to_bids[auc]:
            j = 0
            while j < len(bins) and bins[j] < bid.time:
                j += 1
            if j >= len(bins):
                print j
            if bid.is_human:
                h.append(bins[j])
            elif bid.is_robot:
                r.append(bins[j])

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    plt.setp(ax1.get_xticklabels(), fontsize=10, rotation='vertical')
    plt.setp(ax2.get_xticklabels(), fontsize=10, rotation='vertical')

    ax1.xaxis.set_major_locator(MaxNLocator(len(bins)))
    ax1.set_title("Humans")
    ax1.hist(h, bins, normed=1, facecolor='green', alpha=0.75)

    ax2.xaxis.set_major_locator(MaxNLocator(len(bins)))
    ax2.set_title("Robots")
    ax2.hist(r, bins, normed=1, facecolor='red', alpha=0.75)

    plt.savefig("auctions/bids_to_time_bucket.png",bbox_inches='tight')
    plt.close()


def draw_bids_unique(n):
    aucs = get_random_auctions(n)
    h = []
    r = []
    rngs = [5, 10, 20, 50, 100, 500, 1000]
    for rng in rngs:
        for auc in aucs:
            q = deque()
            d = defaultdict(int)
            u = 0
            for bid in shared.auction_to_bids[auc]:
                if bid.bidder_id not in d or d[bid.bidder_id] == 0:
                    u += 1

                d[bid.bidder_id] += 1
                q.append(bid.bidder_id)

                if len(q) > rng:
                    rid = q.popleft()
                    d[rid] -= 1
                    if not d[rid]:
                        u -= 1

                if bid.is_human:
                    h.append(u)
                elif bid.is_robot:
                    r.append(u)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        plt.setp(ax1.get_xticklabels(), fontsize=10, rotation='vertical')

        ax1.set_title("Humans_%s" % rng)
        bins = rng if rng < 101 else 50
        ax1.xaxis.set_major_locator(MaxNLocator(bins))
        ax1.hist(h, bins, normed=1, facecolor='green', alpha=0.75)

        ax2.set_title("Robots_%s" % rng)
        plt.setp(ax2.get_xticklabels(), fontsize=10, rotation='vertical')
        ax2.xaxis.set_major_locator(MaxNLocator(bins))
        ax2.hist(r, bins, normed=1, facecolor='red', alpha=0.75)

        plt.savefig("auctions/bids_to_unique_%s.png" % rng,bbox_inches='tight')
        plt.close()


def draw_auctions_bids(auction_to_bids, count):
    rcParams['figure.figsize'] = 48, 48
    keys = auction_to_bids.keys()
    idx = [i for i in xrange(len(keys))]
    np.random.shuffle(idx)

    data = dict()
    i = 0
    while i < len(idx):
        if not count:
            break

        auc = keys[idx[i]]

        i += 1
        count -= 1

        data[auc] = [(bid.time, bid.is_human, bid.is_robot) for bid in auction_to_bids[auc]]

    n = len(data)
    X = []
    Y = []

    rc = [1,0,0] # red robots
    hc = [0,0,1] # blue humans
    uc = [0,1,0] # green unknowns
    colors = []
    width = 0.2
    i = 0
    sel = []
    for auc, row in data.iteritems():
        sel.append(auc)
        for j in [0,1,2]:
            # 0 - unknonw
            # 1 - humans
            # 2 - robots
            x = i+j*width+0.3
            i += 1

            c = uc
            if j == 1: # human
                c = hc
            if j == 2: # robot
                c = rc

            for y, is_human, is_robot in row:
                s = 0
                if is_human:
                    s = 1
                elif is_robot:
                    s = 2

                if s != j:
                    continue

                colors.append(c)
                X.append(x)
                Y.append(y)

    fig, ax = plt.subplots()
    offsets = [i*3 for i in xrange(n)]

    def format_date(y, pos=None):
        return datetime.fromtimestamp(y).strftime("%b %d %H:%M:%S")

    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.yaxis.set_major_locator(MaxNLocator(64))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()

    ax.set_xticks(offsets)
    ax.set_xticklabels(sel)
    # plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')

    plt.scatter(X, Y, c=colors, s=20, marker='_')

    plt.savefig("auctions/auctions_to_bids.png", bbox_inches='tight')
    plt.close()

    """
    ms = []
    for row in data:
        n = len(row)
        m = 0
        for i in xrange(1,n):
            if row[i]-row[i-1]>m:
                m = row[i]-row[i-1]
        ms.append(m)

    noz = []
    for i in xrange(len(data)):
        k = sel[i]
        m = ms[i]
        if len(data[i]) > 1:
            noz.append((m,k))

    print sorted(noz)
    """
    """
    for i in xrange(10):
        print sel[i],
    print

    for i in xrange(10):
        print data[i], ',',
    print
    """

# draw_auction_bids(dict(), 1)

def draw_graph(G, bidders):
    plt.figure(1,figsize=(64,64))
    robots = [bidder_id for bidder_id, bidder in bidders.iteritems() if is_robot(bidder)]
    humans = [bidder_id for bidder_id, bidder in bidders.iteritems() if is_human(bidder)]
    pos = nx.spring_layout(G) # positions for all nodes
    print "%s humans and %s robots" % (len(humans), len(robots))
    nx.draw_networkx(G, pos,
        nodelist=robots,
        node_color="r",
        node_size=20,
        alpha=0.5,
        with_labels=False)

    nx.draw_networkx(G, pos,
        nodelist=humans,
        node_color="b",
        node_size=20,
        alpha=0.5,
        with_labels=False)

    plt.axis('off')
    plt.savefig("graph.png")
    plt.close()


def draw_robots_country():
    rcParams['figure.figsize'] = 48, 24
    # TODO: try increments instead of bids with weighted country
    n = len(shared.countries)
    rc = [0 for i in xrange(n)]
    hc = [0 for i in xrange(n)]

    for auc, bids in shared.auction_to_increments.iteritems():
        for bid in bids:
            if bid.is_robot:
                rc[shared.countries.index(bid.country)] += 1
            elif bid.is_human:
                hc[shared.countries.index(bid.country)] += 1

    fig, ax = plt.subplots()
    width = 0.5
    ind = np.arange(n)
    rects1 = ax.bar(ind, hc, width, color='g', alpha=0.7)
    rects2 = ax.bar(ind+width, rc, width, color='r', alpha=0.5)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(shared.countries)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    plt.savefig("country_distribution_using_increments.png", bbox_inches='tight')
    plt.close()



