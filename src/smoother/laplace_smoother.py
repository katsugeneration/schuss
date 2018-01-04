import collections
from util.iterator import flatten


class LaplaceSmoother(object):
    '''
    Add delta smoothing algorithme module
    '''
    def __init__(self, delta=1):
        self.delta = delta

    def smooth(self, counter, words):
        '''
        return P(w_n | w_1, w_2, ... w_n-1) = C(w_n | w_1, w_2, ... w_n-1) + delta / C(w_1, w_2, ... w_n-1) + |V| * delta.
        when C(w_1, w_2, ... w_n-1) is zero, P(w_n | w_1, w_2, ... w_n-2) is used.
        '''
        query = words[:-1]
        dic = counter.search(query)
        if len(dic) == 0:
            return self.smooth(counter, query)

        try:
            b = counter.search(words)
            count = sum(flatten(b)) if isinstance(b, collections.Iterable) else b
        except:
            count = 0
        count += self.delta
        l = list(flatten(dic))
        return count / (sum(l) + self.delta * len(l))
