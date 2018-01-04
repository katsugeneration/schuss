# -*- coding: utf-8 -*-
from util.iterator import sliding_window


class Schuss(object):
    '''
    Spell checker main module.
    '''
    def __init__(self, counter, tokenizer, smoother, distance, window_size=3):
        '''
        :parameters:
            counter: n-gram counter object. Expect counted.
            tokenizer: tokenizer object. SentecePiece or mecab ot etc...
            smoother: probability smoothing object.
            distance: words distance  mesurement object.
            window_size: considering window size before word.
        '''
        self.counter = counter
        self.tokenizer = tokenizer
        self.smoother = smoother
        self.distance = distance
        self.window_size = window_size

    def detect(self, sentence, correct_threshold=0.05):
        '''
        detect sentence miss position.
        :parameters:
            correct_threshold: probability threshold for correct word order judgement.
        '''
        words = self.tokenizer.encode(sentence)
        costs = [0] * len(words)

        for i, s in enumerate(sliding_window(words, self.window_size)):
            p = self.smoother.smooth(self.counter, s)
            if p >= correct_threshold:
                continue
            for j in range(i, i + self.window_size):
                costs[j] += -1

        return words, costs

    def pickup(self, words, costs, num=10, distance=1, cost_threshold=-2, beta=0.5):
        candidates = []
        for i, w in enumerate(words):
            if costs[i] > cost_threshold:
                candidates.append([(w, 0)])
            else:
                c = self._pickup_candidate_item(w, distance)
                if (w, 0) not in c:
                    c += [(w, 0)]
                candidates.append(c)

        cs = [("", 0.0)]

        def _calc(args):
            c, s = args
            sentence = s[0] + c[0]
            rate = s[1] \
                + (beta ** (c[1] + 1)) \
                + self.smoother.smooth(self.counter, sentence[-self.window_size:])
            return (sentence, rate)

        for candidate in candidates:
            before_cs = cs
            cs = []

            def items():
                for c in candidate:
                    for s in before_cs:
                        yield (c, s)

            cs = map(_calc, items())
            cs = sorted(cs, key=lambda r: r[1], reverse=True)[:num]
        return cs

    def _pickup_candidate_item(self, word, distance):
        return list(map(lambda w: (w, self.distance.measure(word, w)),
                    filter(lambda w: self.distance.measure(word, w) <= distance,
                    self.tokenizer.vocab)))

    def fix(self, sentence):
        pass
