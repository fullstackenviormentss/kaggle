import math
import csv
from utils import skip, time_to_hour
from collections import defaultdict
from collections import namedtuple
import matplotlib.pyplot as plt
import pylab as P
from datetime import datetime, timedelta
from bids import Bid

country_to_time = defaultdict(list)
rbo_country_to_time = defaultdict(list)

def read_bids():
	with open('clean_bids.csv', 'rb') as bidsfile:
		reader = csv.reader(bidsfile, delimiter=',', quotechar='|')
		skip(reader, 1)
		c = 0
		prev_bid = None
		for row in reader:
			c += 1
			if c % 1e6 == 0:
				print "Read bids progress: %s" % c

			bid = Bid(*row)
			ct = None
			if bid.is_human:
				ct = country_to_time
			if bid.is_robot:
				ct = rbo_country_to_time
			if ct is None:
				continue

            u = float(52631.578)
            t = bid.time
            d = datetime.today()
            td = timedelta(seconds=(t/u))
			dt = d + td
			ct[bid.country].append(dt.hour)

		print "Bids read done %s" % len(country_to_time)

read_bids()

bins = [i for i in xrange(24)]
with open('country_to_time.csv', 'wb') as result:
	fieldnames = ['country'] + [str(i) for i in xrange(24)]
	writer = csv.DictWriter(result, fieldnames=fieldnames)
	writer.writeheader()
	for c, times in country_to_time.iteritems():
		x = [0.0 for i in xrange(24)]
		for t in times:
			x[t] += 1

		s = sum(x)
		x = [float(i)/s for i in x]

		# P.figure(num=c)
		# P.axis([0, 23, 0, 1])
		# print c, times
		# raw_input()
		P.hist(times, bins, histtype='bar', facecolor='green', rwidth=0.8, alpha=0.7)
		if rbo_country_to_time[c]:
			P.hist(rbo_country_to_time[c], bins, histtype='bar', facecolor='red', rwidth=0.8, alpha=0.5)
		# P.show()
		P.savefig('graphs/time_per_country/' + c + '.png')
		P.close()


		d = dict()
		d["country"] = c
		for i in xrange(24):
			d[str(i)] = x[i]

		writer.writerow(d)


