
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
        ret = counter.search(query)
        if len(ret) == 0:
            return self.smooth(counter, query)

        count = ret[tuple(words)] if tuple(words) in ret else 0
        count += self.delta
        return count / (sum(ret.values()) + self.delta * len(ret))
