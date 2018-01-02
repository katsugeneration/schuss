# -*- coding: utf-8 -*-
from util.sliding import sliding_window


class LossyCountingNGram(object):
    '''
    Count n-gram using Lossy Counting algorithme

    :members:
        _symbol_num: symbol count in training corpus.
        _items: item dictionary. key is symbol which join \'@\' n-gram. value is symbol count.
    '''
    def __init__(self, window_size=3, epsilon=1e-5):
        '''
        :parameters:
            window_size: n in n-gram.
            epsilon: Lossy Counting algorithme quality parameter
        '''
        self._symbol_num = 0
        self._buckets_num = 0
        self.epsilon = epsilon
        self.window_size = window_size
        self._items = {}

    def fit(self, X, y=None):
        self._count_ngram(X)

    def search(self, X):
        '''
        Search start query n-gram

        :parameters:
            X: query array-like object
        '''
        query = "@".join(X)
        ret = {}
        if query == "":
            dic = self._items.items()
        else:
            dic = filter(lambda x: (x[0].startswith(query+"@") or x[0] == query), self._items.items())

        for key, value in dic:
            ret[tuple(key.split("@"))] = value
        return ret

    def _count_ngram(self, X):
        self._symbol_num = 0
        self._buckets_num = 0

        for line in X:
            if line.strip() == "":
                continue
            line = line.strip().split()
            self._count_symbols(line)

    def _count_symbols(self, arr):
        for s in sliding_window(arr, self.window_size):
            self._symbol_num += 1
            symbol = "@".join(s)

            if symbol in self._items:
                self._items[symbol] += 1
            else:
                self._items[symbol] = self._buckets_num + 1

            if self._symbol_num % int(1 / self.epsilon) == 0:
                self._buckets_num += 1
                self._items = self._remove_items(self._items, self._buckets_num)

    def _remove_items(self, items, threshold):
        ret = {}
        for key, value in filter(lambda x: x[1] >= threshold, items.items()):
            ret[key] = value
        return ret
