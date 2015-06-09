import numpy as np
import operator
import math
import bisect
from collections import defaultdict, namedtuple, deque

import shared
import utils
from bids import Bid

class Interval:
    def __init__(self, start):
        self.start = start
        self.end = start

# namedtuple('StoredIncrement', ['auction', 'bidder_id', 'time', 'price', 'diff_price', 'is_human', 'is_robot', 'is_last', 'merchandise', 'ips_count', 'country'])
class StoredIncrement:
    def __init__(self, *args):
        self.auction = args[0]
        self.bidder_id = args[1]
        self.time = int(args[2])  # time
        self.price = int(args[3])  # price
        self.diff_price = int(args[4])  # diff_price
        self.is_human = args[5] == 'True'  # is_human
        self.is_robot = args[6] == 'True'  # is_robot
        self.is_last = args[7] == 'True'  # is_last
        self.merchandise = args[8]
        self.ips_count = int(args[9])  # ips_count
        self.country = args[10]


class Increment:
    def __init__(self):
        self.bids = []
        self.frequent_country = ''

    def min_price(self):
        return self.bids[0].price

    def max_price(self):
        return self.bids[-1].price

    def bids_count(self):
        return len(self.bids)

    def countries_count(self):
        return len(set([bid.country for bid in self.bids]))

    def ref_count(self):
        return len(set([bid.url for bid in self.bids]))

    def start_time(self):
        return self.bids[0].time

    def end_time(self):
        return self.bids[-1].time

    @property
    def auction(self):
        return self.bids[0].auction

    @property
    def bidder_id(self):
        return self.bids[0].bidder_id

    @property
    def time(self):
        return self.start_time()

    @property
    def price(self):
        return self.max_price()

    @property
    def diff_price(self):
        return self.max_price() - self.min_price()

    @property
    def is_human(self):
        return self.bids[0].is_human

    @property
    def is_robot(self):
        return self.bids[0].is_robot

    @property
    def is_last(self):
        return self.bids[-1].is_last

    @property
    def is_win(self):
        return self.bids[-1].is_win

    @property
    def merchandise(self):
        return self.bids[0].merchandise

    @property
    def ips_count(self):
        return len(set([bid.ip for bid in self.bids]))

    @property
    def country(self):
        c = defaultdict(int)
        if not self.frequent_country:
            for bid in self.bids:
                c[bid.country] += 1
            self.frequent_country = max(c.iteritems(), key=operator.itemgetter(1))[0] if len(c) else ''

        return self.frequent_country


class Bidder:
    def __init__(self, bidder_id, payment_account, address, outcome=None):
        self.bidder_id = bidder_id
        self.payment_account = payment_account
        self.address = address
        if outcome is not None:
            self.outcome = float(outcome)
        else:
            self.outcome = None

        self.auctions = defaultdict(list)

        self.devices = set()
        self.devices_per_auction = defaultdict(set)

        self.products = defaultdict(int)

        self.bids = []
        self.increments_per_auction = defaultdict(list)
        self.last_bids = []
        self.win_bids = []

        # self.resp_times_auction = []  # resp times within auction
        # self.resp_times = []  # resp time in a system to others bids

        self.countries = defaultdict(int)
        self.counties_per_auction = defaultdict(set)

        self.referrals = defaultdict(int)
        self.referrals_per_auction = defaultdict(set)

        self.ips = defaultdict(int)
        self.ips_per_auction = defaultdict(set)

        self.intervals = dict()
        # self.time_prob_cum = 0.0

    def get_addr_1(self):
        return self.address[:32]

    def get_addr_2(self):
        return self.address[32:]

    def get_payment_type(self):
        return self.payment_account[:32]

    def get_payment_acct(self):
        return self.payment_account[32:]

    def get_frequent(self, arr):
        return max(arr.iteritems(), key=operator.itemgetter(1))[0] if len(arr) else ''

    def get_sorted_frequent(self, arr):
        a = [(v, k) for k,v in arr.iteritems()]
        a.sort(reverse=True)
        return a

    @property
    def is_robot(self):
        return utils.is_robot(self)

    @property
    def is_human(self):
        return utils.is_human(self)

    def get_per_auction(self, func, arr):
        per_auction = []
        for k,v in arr.iteritems():
            per_auction.append(len(v))

        return func(per_auction) if per_auction else -1

    def get_sim_auctions(self):
        I = []
        for a, interval in self.intervals.iteritems():
            I.append((interval.start, 0))
            I.append((interval.end, 1))

        I.sort()
        sim = 0
        max_sim = 0

        for i in xrange(len(I)):
            time, kind = I[i]
            if kind == 0:  # is start
                sim += 1
            else:
                sim -= 1

            if max_sim < sim:
                max_sim = sim

        return max_sim

    def get_price_rmse(self, last_bids, all_prices, attr):
        # Root Mean Squared Error (RMSE)
        prices = defaultdict(list)
        for bid in last_bids:
            k = getattr(bid, attr)
            prices[k].append(bid)

        rmse = 0.0
        n = 0
        for k, bids in prices.iteritems():
            for bid in bids:
                rmse += math.pow(all_prices[k]-bid.price, 2)
            n += len(bids)
        if n:
            rmse /= float(n)

        return math.sqrt(rmse)

    def get_all_increments(self):
        all_increments = []
        for auc, increments in self.increments_per_auction.iteritems():
            all_increments += increments

        return all_increments

    def find_increment_patterns(self):
        var = []
        for auc, increments in self.increments_per_auction.iteritems():
            prev_time = None
            d = []
            if len(increments) < 10:
                continue
            for inc in increments:
                if not prev_time:
                    prev_time = inc.time
                else:
                    d.append(inc.time-prev_time)

            var.append(np.var(d))

        if self.is_human:
            print 'human',
        if self.is_robot:
            print 'robot',
        print 'variance: ', ['%.2f' % v for v in var]
        print

    def find_inc_price_patterns(self):
        var = []
        for auc, increments in self.increments_per_auction.iteritems():
            if len(increments) < 10:
                continue
            var.append(np.var([i.diff_price for i in increments]))

        if self.is_human:
            print 'human',
        if self.is_robot:
            print 'robot',
        print 'price variance: ', ['%.2f' % v for v in var]
        print

    def get_ip_class(self, ip):
        octets = ip.split('.')
        if not octets or not octets[0]:
            return ''
        fo = int(octets[0])
        if fo <= 126:
            return 'A'
        elif fo == 127:
            return 'localhost'
        elif fo > 127 and fo <= 191:
            return 'B'
        elif fo > 191 and fo <= 223:
            return 'C'
        elif fo > 223 and fo <= 239:
            return 'D'
        else:
            return 'E'

    def get_ip_rank(self, bids):
        ip_rank = 0.0
        c = 0
        for bid in bids:
            if bid.ip_pref not in shared.ip_to_bidder:
                continue
            h, r = shared.ip_to_bidder[bid.ip_pref]
            if (h+r) == 0:
                continue
            c += 1
            ip_rank += float(r) / (h+r)
        if c:
            return ip_rank/c
        return 0.0

    def get_change_rate(self, prop):
        last_c = None
        change_rate = 0
        for bid in self.bids:
            if last_c is None:
                last_c = getattr(bid, prop)
            elif last_c != getattr(bid, prop):
                last_c = getattr(bid, prop)
                change_rate += 1

        if self.bids:
            return float(change_rate)/len(self.bids)
        return 0.0

    def get_strategy(self, auc):
        bids = shared.auction_to_bids[auc]
        n_short = 30
        q_short = deque()  # bids within last n_short seconds

        n_long = 2500
        q_long = deque()

        x_short_agg = int(n_short * 0.33)
        x_short = 0  # number of bids made by this user in short period of time
        p_short_agg = 0  # probability that user plays aggressively in short term

        x_long_agg = int(n_long * 0.25)
        x_long = 0  # number of bids made by this user in long period of time
        p_long_agg = 0  # probability that user plays aggressively in long term

        start = False
        count = 0

        def pque(queue, bid, n):
            queue.append(bid)
            t = bid.time
            d = 1 if bid.bidder_id == self.bidder_id else 0
            while queue and t - queue[0].time > n:
                b = qeueu.popleft()
                if b.bidder_id == self.bidder_id:
                    d -= 1
            return d

        for bid in bids:
            x_short += pque(q_short, bid, n_short)
            x_long += pque(q_long, bid, n_long)

            if not start and bid.bidder_id == self.bidder_id:
                start = True

            if not start:  # user has not started bidding
                continue

            count += 1

            if x_short >= x_short_agg:
                p_short_agg += 1

            if x_long >= x_long_agg:
                p_long_agg += 1

            # stop counting when user already left auction
            if bid.bidder_id == self.bidder_id and bid.is_last:
                start = False

        p_short_agg = float(p_short_agg)/count
        p_long_agg = float(p_long_agg)/count

        print "Bidder %s in auction %s attempted strategy: short_aggressive (%.2f), long_aggressive (%.2f)" % (self.bidder_id, auc, p_short_agg, p_long_agg)


    def features(self):
        # global human_cnt
        # if self.is_robot or (self.is_human and human_cnt > 0):
        #     if self.is_human:
        #         human_cnt -= 1
        #     self.find_increment_patterns()
            #self.find_inc_price_patterns()

        increments = self.get_all_increments()
        bids_count = len(self.bids)
        true_bids_count = len(increments)

        # Auctions
        auctions_count = len(self.auctions)
        sim_auctions = self.get_sim_auctions()
        won_auctions_count = len(self.win_bids)
        auction_rank = 0.0
        for auc, bids in self.auctions.iteritems():
            auction_rank += shared.auction_rank[auc]
        if self.auctions:
            auction_rank /= len(self.auctions)

        # Price
        human_price_rmse_per_auction = self.get_price_rmse(self.last_bids, shared.human_median_price_per_auction, 'auction')
        human_price_rmse_per_product = self.get_price_rmse(self.last_bids, shared.human_median_price_per_product, 'merchandise')

        robot_price_rmse_per_auction = self.get_price_rmse(self.last_bids, shared.robot_median_price_per_auction, 'auction')
        robot_price_rmse_per_product = self.get_price_rmse(self.last_bids, shared.robot_median_price_per_product, 'merchandise')

        # Stats per auction
        avg_countries_per_auction = self.get_per_auction(np.average, self.counties_per_auction)
        median_countries_per_auction = self.get_per_auction(np.median, self.counties_per_auction)
        std_countries_per_auction = self.get_per_auction(np.std, self.counties_per_auction)

        avg_devices_per_auction = self.get_per_auction(np.average, self.devices_per_auction)
        median_devices_per_auction = self.get_per_auction(np.median, self.devices_per_auction)
        std_devices_per_auction = self.get_per_auction(np.std, self.devices_per_auction)

        avg_referrals_per_auction = self.get_per_auction(np.average, self.referrals_per_auction)
        median_referrals_per_auction = self.get_per_auction(np.median, self.referrals_per_auction)
        std_referrals_per_auction = self.get_per_auction(np.std, self.referrals_per_auction)

        avg_ips_per_auction = self.get_per_auction(np.average, self.ips_per_auction)
        median_ips_per_auction = self.get_per_auction(np.median, self.ips_per_auction)
        std_ips_per_auction = self.get_per_auction(np.std, self.ips_per_auction)

        # avg_inc_per_auction = self.get_per_auction(np.average, self.increments_per_auction)
        # median_inc_per_auction = self.get_per_auction(np.median, self.increments_per_auction)
        # std_inc_per_auction = self.get_per_auction(np.std, self.increments_per_auction)

        # IP
        avg_ips_per_increment = utils.divide(sum([inc.ips_count for inc in increments]), len(increments))
        frequent_ip = '.'.join(self.get_frequent(self.ips).split('.')[:2])
        ip_octets = defaultdict(int)

        for ip, count in self.ips.iteritems():
            octets = ip.split('.')
            ip_octets['.'.join(octets[:1])] += count
            ip_octets['.'.join(octets[:2])] += count
            ip_octets['.'.join(octets[:3])] += count
            ip_octets['.'.join(octets[:4])] += count

        sorted_octets = sorted(ip_octets.items(), key=operator.itemgetter(1), reverse=True)

        # Countries
        cnt_mask = 0
        seen_countries = set(self.countries.keys())
        for country in seen_countries:
            i = shared.countries.index(country)
            cnt_mask |= (1<<(i+1))
        cnt_mask = str(cnt_mask)

        countries_inc = defaultdict(int)
        for inc in self.bids:
            countries_inc[inc.country] += 1

        country_rank = 0
        for country, count in countries_inc.iteritems():
            country_rank += shared.country_rank[country] * count
        if len(countries_inc):
            country_rank = float(country_rank) / sum(countries_inc.values())

        regions_mask = 0
        seen_regions = set([shared.country_to_region[c] for c in seen_countries])
        for region in seen_regions:
            i = shared.regions.index(region)
            regions_mask |= (1<<(i+1))
        regions_mask = str(regions_mask)

        # Products
        all_products = ['mobile', 'jewelry', 'home goods', 'sporting goods', 'auto parts', 'office equipment', 'computers', 'books and music', 'furniture', 'clothing']
        products_mask = 0  # compute all products user bidded on as a number
        for p in self.products.keys():
            i = all_products.index(p)+1
            if not i:
                continue
            products_mask |= (1<<i)
        products_mask = str(products_mask)

        # Bids
        bid_on_unpopular = -1.0
        for bid in self.bids:
            if bid.merchandise in ["auto parts", "clothing", "furniture"]:
                bid_on_unpopular = 1.0
                break

        # Generate
        # TODO: add count of times user reached max price in auction (measure greediness)
        labels = ["country", "device", "product", "ip", "ref", "auction", "bids", "increments", "won_auctions", "sim_auctions"]
        values = [len(self.countries), len(self.devices), len(self.products), len(self.ips), len(self.referrals), auctions_count, len(self.bids), true_bids_count, won_auctions_count, sim_auctions]
        #ops = ["+", "-", "*", "/"]
        ops = ["/"]
        generated_features = dict()

        for op in ops:
            for i in xrange(len(labels)):
                al = labels[i]
                av = values[i]
                for j in xrange(i+1, len(labels)):
                    bl = labels[j]
                    bv = values[j]
                    if op == "+":
                        generated_features[al+op+bl] = av+bv
                    elif op == "-":
                        generated_features[al+op+bl] = av-bv
                    elif op == "*":
                        generated_features[al+op+bl] = av*bv
                    elif op == "/":
                        generated_features[al+op+bl] = utils.divide(av,bv)

        # Time
        time_hist_prob = 0
        for bid in self.bids:
            i = bisect.bisect_left(shared.human_hist_bins, bid.time)
            time_hist_prob += shared.human_hist[i-1]

        if self.bids:
            time_hist_prob /= float(len(self.bids))


        features = {
            "set_0": self.bidder_id in shared.set0,
            # Address
            # "addr_1": self.get_addr_1(),
            # "addr_2": self.get_addr_2(),
            # "is_addr_1_unique": len(shared.addr_1[self.get_addr_1()]) == 1,
            # "is_addr_2_unique": len(shared.addr_2[self.get_addr_2()]) == 1,
            # "addr_1=a3d2de7675556553a5f08e4c88d2c228": self.get_addr_1() == 'a3d2de7675556553a5f08e4c88d2c228',

            # Devices
            # "devices": len(self.devices),
            # TODO: devices per last X bids
            #"avg_devices_per_auction": avg_devices_per_auction,
            #"median_devices_per_auction": median_devices_per_auction,
            #"std_devices_per_auction": std_devices_per_auction,

            # Time
            "time_hist_prob": time_hist_prob,

            # Country
            #"unique_countries_count": len(self.countries),
            # "frequent_country": self.get_frequent(self.countries),
            "avg_countries_per_auction": avg_countries_per_auction,
            "median_countries_per_auction": median_countries_per_auction,
            "std_countries_per_auction": std_countries_per_auction,
            # "seen_countries": cnt_mask,
            #"country_rank": country_rank,
            # "seen_regions": regions_mask,
            #"regions_count": len(seen_regions),
            "country_change": self.get_change_rate('country'),

            # Products
            # "frequent_product": self.get_frequent(self.products),
            "bid_on_unpopular": bid_on_unpopular,
            # "products_mask": products_mask,

            # IP
            #"unique_ips": len(self.ips),
            "avg_ips_per_auction": avg_ips_per_auction,
            "median_ips_per_auction": median_ips_per_auction,
            "std_ips_per_auction": std_ips_per_auction,
            #"avg_unique_ips": utils.divide(len(self.ips), bids_count),
            #"avg_ips": utils.divide(sum(self.ips.values()), bids_count),
            "avg_ips_per_increment": avg_ips_per_increment,
            # "frequent_ip": frequent_ip,
            #"frequent_ip_class": self.get_ip_class(frequent_ip),  # http://www.vlsm-calc.net/ipclasses.php
            # "most_popular_octet": sorted_octets[0][0] if sorted_octets else '',
            #"ip_rank": self.get_ip_rank(self.bids),
            "ip_change": self.get_change_rate('ip_pref'),

            # Referrals
            # "frequent_referral": self.get_frequent(self.referrals),
            #"referrals_count": len(self.referrals),
            "avg_referrals_per_auction": avg_referrals_per_auction,
            "median_referrals_per_auction": median_referrals_per_auction,
            "std_referrals_per_auction": std_referrals_per_auction,
            "referral_change": self.get_change_rate('url'),

            # Auctions
            # "auctions_count": auctions_count,
            #"won_auctions_count": won_auctions_count,
            #"sim_auctions": sim_auctions,
            # "auction_rank": auction_rank,

            # Payment
            # "payment_type": self.get_payment_type(),
            # "payment_acct": self.get_payment_acct(),
            # "is_pmt_type_unique": len(shared.pmt_type[self.get_payment_type()]) == 1,
            # "is_pmt_acct_unique": len(shared.pmt_accnt[self.get_payment_acct()]) == 1,
            #"payment_type=addr_1": self.get_payment_type() == self.get_addr_1(),
            #"payment_acct=addr_2": self.get_payment_acct() == self.get_addr_2(),

            # Bids
            #"true_bids_count": true_bids_count,
            # "avg_inc_per_auction": avg_inc_per_auction,
            # "median_inc_per_auction": median_inc_per_auction,
            # "std_inc_per_auction": std_inc_per_auction,

            # Price
            # rmse is calculated based on won_price and measure user price threshold/estimate
            # expect to have it bigger for humans and smaller for robots
            #"human_price_rmse_per_auction": human_price_rmse_per_auction,
            #"human_price_rmse_per_auction": human_price_rmse_per_auction,

            #"robot_price_rmse_per_auction": robot_price_rmse_per_auction,
            #"robot_price_rmse_per_product": robot_price_rmse_per_product,
        }

        features.update(generated_features)
        # features.update(buckets_dict)

        return features

    def update(self, bid, time_prob):
        if bid.auction not in self.auctions:
            self.products[bid.merchandise] += 1

        self.auctions[bid.auction].append(bid)
        self.devices.add(bid.device)
        self.devices_per_auction[bid.auction].add(bid.device)

        self.ips[bid.ip] += 1
        self.ips_per_auction[bid.auction].add(bid.ip)

        self.countries[bid.country] += 1
        self.counties_per_auction[bid.auction].add(bid.country)

        self.referrals[bid.url] += 1
        self.referrals_per_auction[bid.auction].add(bid.url)

        self.bids.append(bid)
        # Record first and last bid of this bidder in particular auction
        if bid.auction not in self.intervals:
            self.intervals[bid.auction] = Interval(bid.time)
        else:
            self.intervals[bid.auction].end = bid.time

        # self.time_prob_cum += time_prob

        if bid.is_last:
            self.last_bids.append(bid)

        if bid.is_win:
            self.win_bids.append(bid)


