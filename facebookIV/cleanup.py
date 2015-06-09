import csv
import sys
import operator
import numpy as np
from collections import defaultdict
from collections import namedtuple
from utils import skip
from bidder import Bidder, Interval
import utils
import shared

bids = []
bidders = dict()

# Bid = namedtuple('Bid',  'bid_id bidder_id auction merchandise device time country ip url')
class Bid:
    def __init__(self, bid_id, bidder_id, auction, merchandise, device, time, country, ip, url):
        self.bid_id = bid_id
        self.bidder_id = bidder_id
        self.auction = auction
        self.merchandise = merchandise
        self.device = device
        self.time = int(time)
        # remove notion of days in our data set
        for i, day in enumerate(shared.time_boundaries):
            s, e = day[0], day[1]
            if self.time >= s and self.time <= e:
                self.time -= s
                self.auction = '%s_%s' % (self.auction, i)
                break
        self.country = country
        self.ip = ip
        self.url = url

        self.order = 0
        self.price = 0
        self.max_price = 0
        self.won_price = 0
        self.is_last = False
        self.is_win = False
        self.is_robot = False
        self.is_human = False


    def __repr__(x):
        return str({
                'bid_id': x.bid_id,
                'bidder_id': x.bidder_id,
                'auction': x.auction,
                'merchandise': x.merchandise,
                'device': x.device,
                'time': x.time,
                'country': x.country,
                'ip': x.ip,
                'url': x.url,
                'order': x.order,
                'price': x.price,
                'max_price': x.max_price,
                'won_price': x.won_price,
                'is_last': x.is_last,
                'is_win': x.is_win,
                'is_robot': x.is_robot,
                'is_human': x.is_human
            })


def read_bidders(filename):
    with open('clean_%s' % filename, 'wb') as cbfile:
        fieldnames = ['bidder_id', 'pmt_type', 'pmt_account', 'addr_1', 'addr_2']
        if 'train' in filename:
            fieldnames.append('outcome')
        writer = csv.DictWriter(cbfile, fieldnames=fieldnames)
        writer.writeheader()
        with open(filename, 'rb') as bidsfile:
            reader = csv.reader(bidsfile,  delimiter=',',  quotechar='|')
            skip(reader, 1)
            for row in reader:
                b = Bidder(row[0], row[1], row[2], row[3] if len(row) > 3 else None)
                bidders[b.bidder_id] = b
                mrow = {
                    'bidder_id': b.bidder_id,
                    'pmt_type': b.get_payment_type(),
                    'pmt_account': b.get_payment_acct(),
                    'addr_1': b.get_addr_1(),
                    'addr_2': b.get_addr_2(),
                }
                if 'train' in filename:
                    mrow['outcome'] = b.outcome
                writer.writerow(mrow)


def read_bids():
    with open('bids.csv', 'rb') as bidsfile:
        reader = csv.reader(bidsfile, delimiter=',', quotechar='|')
        skip(reader, 1)
        c = 0
        prev_bid = None
        for row in reader:
            c += 1
            if c % 1e6 == 0:
                print "Read bids progress: %s" % c

            bid = Bid(*row)
            bids.append(bid)

        print "Bids read done: %s" % len(bids)


def write_bids(bids):
    print "Writing results"
    c = 0
    with open('clean_bids.csv', 'wb') as result:
        fieldnames = ['bid_id','bidder_id','auction','merchandise','device','time','country','ip','url', 'order', 'price', 'max_price', 'won_price', 'is_last', 'is_win', 'is_robot', 'is_human']
        writer = csv.DictWriter(result, fieldnames=fieldnames)
        writer.writeheader()
        for x in bids:
            c += 1
            if c % 1e6 == 0:
                print "Write bids progress: %s" % c
            writer.writerow({
                'bid_id': x.bid_id,
                'bidder_id': x.bidder_id,
                'auction': x.auction,
                'merchandise': x.merchandise,
                'device': x.device,
                'time': x.time,
                'country': x.country,
                'ip': x.ip,
                'url': x.url,
                'order': x.order,
                'price': x.price,
                'max_price': x.max_price,
                'won_price': x.won_price,
                'is_last': x.is_last,
                'is_win': x.is_win,
                'is_robot': x.is_robot,
                'is_human': x.is_human
            })


read_bidders('train.csv')
read_bidders('test.csv')
read_bids()

bids.sort(key=lambda x: (x.time, x.bidder_id, x.auction))
print "Done sorting"

auction = ''
merchandise = ''
auction_to_merch = dict()
auction_to_bids = defaultdict(list)
ip_to_country = defaultdict(list)
octet_to_country = defaultdict(list)

bids_without_country = []

user_to_bid_count = defaultdict(int)
prev_bid = None
sim = set()

# Fix country
for bid in bids:
    if bid.auction not in auction_to_merch:
        auction_to_merch[bid.auction] = dict()
    auction_to_merch[bid.auction][bid.bidder_id] = bid.merchandise
    auction_to_bids[bid.auction].append(bid)
    if bid.country:
        ip_to_country[bid.ip].append(bid.country)
        octets = bid.ip.split('.')
        for i in xrange(1, 5):
            prefix = '.'.join(octets[:i])
            octet_to_country[prefix].append(bid.country)
    else:
        bids_without_country.append(bid)

    if bid.bidder_id in bidders:
        out = bidders[bid.bidder_id].outcome
        bid.is_robot = out is not None and out > 0.1
        bid.is_human = out is not None and out < 0.1

    user_to_bid_count[bid.bidder_id] += 1

# Calculate probability of each octet to belong to some country
ip_prefix_to_country = defaultdict(list)
for octet, countries in octet_to_country.iteritems():
    d = defaultdict(int)
    for c in countries:
        d[c] += 1
    t = len(countries)
    p = [(float(i)/t, country) for country, i in d.iteritems()]
    ip_prefix_to_country[octet] = sorted(p, reverse=True)

print "Found %s bids without country" % len(bids_without_country)
missed_c = 0
for bid in bids_without_country:
    octets = bid.ip.split('.')
    m_p, m_c, m_pr = 0, '', ''
    for i in xrange(4, 0, -1):
        prefix = '.'.join(octets[:i])
        if prefix in ip_prefix_to_country:
            p, c = ip_prefix_to_country[prefix][0]
            if p > m_p:
                m_p = p
                m_c = c
                m_pr = prefix
                break

    if m_p > 0:
        # print "%s to ip %s with probability %.2f based on octet %s" % (m_c, bid.ip, m_p, m_pr)
        bid.country = c
    if not bid.country:
        missed_c += 1
        print "Couldn't find good country for ip %s" % bid.ip

print "Unable to find countries for %s ips" % missed_c

# vote on products for each auction
for k,v in auction_to_merch.iteritems():
    country_votes = v.values()
    d = defaultdict(int)
    for vote in country_votes:
        d[vote] += 1
    m = max(d.iteritems(), key=operator.itemgetter(1))[0] if len(d) else ''
    c = (d[m]+0.0)/sum(d.values())
    auction_to_merch[k] = m
    # if len(d) != 1:
    #     print "%s has %s products, but selected %s, conf %s (%s out of %s)" % (k, len(d), m, c, d[m], sum(d.values()))


# set prices on bids
c = 0
for auc, auc_bids in auction_to_bids.iteritems():
    bidder_to_bids = defaultdict(list)
    max_price = 0
    for i, bid in enumerate(auc_bids):
        c += 1
        if c % 1e6 == 0:
            print "Transform bids progress: %s" % c

        bidder_to_bids[bid.bidder_id].append(bid)
        max_price = max(max_price, len(bidder_to_bids[bid.bidder_id]))

        bid.order = i

        # fix merchandise
        bid.merchandise = auction_to_merch[auc]

    for bidder_id, user_bids in bidder_to_bids.iteritems():
        for i, bid in enumerate(user_bids):
            bid.price = i+1  # how many bids this user actually spent
            bid.max_price = max_price
        user_bids[-1].is_last = True  # indicate last bid from this user in this auction

    for bid in auc_bids:
        bid.won_price = auc_bids[-1].price

    auc_bids[-1].is_win = True

print "Total auctions %s, total bids: %s", (len(auction_to_merch), len(bids))


write_bids(bids)


# idiot e90e4701234b13b7a233a86967436806wqqw4
# 858 auctions (from 15051 total), on average was bidding 1
# when was bidding > 1 was bidding in both categories jewelry and sporting goods

# 70 unknown with 0 bids
# 30 human with 0 bids
# 755 unknown with 1 bid
# 5 robot with 1 bid
# 297 human with 1 bid
