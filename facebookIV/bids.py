import shared
import utils

class Bid:
    def __init__(self, bid_id, bidder_id, auction, merchandise, device, time, country, ip, url, order, price, max_price, won_price, is_last, is_win, is_robot, is_human):
        self.bid_id = bid_id
        self.bidder_id = bidder_id
        self.auction = auction
        self.merchandise = merchandise
        self.device = device
        self.time = int(time)/52631578000
        if country == 'gb':
            country = 'uk'
        self.country = country
        self.ip = ip
        self.url = url

        self.order = int(order)
        self.price = int(price)
        self.max_price = int(max_price)
        self.won_price = int(won_price)

        # indicate if this is the last bid in auction among all users
        self.is_win = is_win == 'True'
        # indicate last bid from this user in this auction
        self.is_last = is_last == 'True'
        self.is_robot = is_robot == 'True'
        self.is_human = is_human == 'True'

        self.outcome = None
        if self.is_robot:
            self.outcome = 1
        if self.is_human:
            self.outcome = 0

        self.prev_unique_50 = 0
        self.prev_unique = 0

        self.time_from_start = 0
        self.time_to_end = 0
        self.time_to_prev_bid = 0
        self.auc_length = 0

    def features(self):
        return {
            # Device
            #'device': self.device,

            # Product
            #'product': self.merchandise,

            #Country
            'country': self.country,

            # Price
            # 'price': self.price,
            'price_fr': utils.short_float(utils.divide(self.price, self.last_price)),

            # IP
            #'ip_3oct': self.ip_3oct,
            #'ip_2oct': self.ip_2oct,

            # Time
            # 'time': self.time,
            # 'time_from_start': self.time_from_start,
            # 'time_to_end': self.time_to_end,
            # format _perc to have 2 digist after .
            'time_from_start_perc': utils.short_float(utils.divide(self.time, self.auc_length)),
            'time_to_end_perc': utils.short_float(utils.divide(self.time_to_end, self.auc_length)),
            # 'time_from_day_start': self.time_from_day_start,
            # 'time_to_prev_bid': self.time_to_prev_bid,

            # Order
            #'is_first': self.is_first,
            #'is_last': self.is_last,
            #'day': self.day,
            'unique': self.prev_unique,
            'unique_50': self.prev_unique_50,
            # how many immediate prev bids are made by the same user?

            # Url
            #'ref': self.url,
        }

    @property
    def ip_pref(self):
        return utils.get_ip_pref(self.ip)

    @property
    def ip_2oct(self):
        return utils.get_ip_part(self.ip, 2)

    @property
    def ip_3oct(self):
        return utils.get_ip_part(self.ip, 3)
