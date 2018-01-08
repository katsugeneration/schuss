# -*- coding: utf-8 -*-
import math
from util.iterator import sliding_window


class Schuss(object):
    '''
    Spell checker main module.
    '''
    def __init__(self, counter, tokenizer, smoother, distance):
        '''
        :parameters:
            counter: n-gram counter object. Expect counted.
            tokenizer: tokenizer object. SentecePiece or mecab ot etc...
            smoother: probability smoothing object.
            distance: words distance  mesurement object.
        '''
        self.counter = counter
        self.tokenizer = tokenizer
        self.smoother = smoother
        self.distance = distance

        self.tokenizer.vocab = list(set(self.tokenizer.vocab) | set(self.counter.vocab))

    def detect(self, sentence, correct_threshold=0.05):
        '''
        detect sentence miss position.
        :parameters:
            correct_threshold: probability threshold for correct word order judgement.
        :output:
            words: target sentence's words(token)
            costs: target words cost considered miss spell
        '''
        words = self.tokenizer.encode(sentence)
        costs = [0] * len(words)

        for i, w in enumerate(words):
            if w not in self.tokenizer.vocab:
                costs[i] += -1

        for i, s in enumerate(sliding_window(words, self.counter.window_size)):
            p = self.smoother.smooth(self.counter, s)
            if p >= correct_threshold:
                continue
            for j in range(i, i + self.counter.window_size):
                costs[j] += -1

        return words, costs

    def pickup(self, words, costs, num=10, distance=1, cost_threshold=-2, beta=0.1):
        '''
        Pickup cnadidate sentences for miss spell fix by likelihood.
        P(C|W)(likelihood) = P(W|C)P(C)
        P(W|C) is probability of misstake. it is considered to distance between C and W
        P(C) is probability of C in language. it is approximate to P(C_i|C_i-1, C_i-2)

        :parameters:
            words: target sentence's words(token)
            costs: target words cost considered miss spell
            num: output numbers
            distance: max distance between taraget word and fixed candidate word
            cost_threshold: target word's min cost for comparing candidate words
            beta: distance cost. P(w_i|c_i) = beta ** distance

        :output:
            candidate sentences array. a candidate is tuple of candidate sentence and likelihood
        '''
        candidates = []
        for i, w in enumerate(words):
            if costs[i] > cost_threshold:
                candidates.append([(w, 0)])
            else:
                c = self._pickup_candidate_item(w, distance)
                if (w, 0) not in c:
                    c += [(w, 0)]
                candidates.append(c)

        cs = [([""], 0.0)]

        def _calc(args):
            c, s = args
            sentence = (s[0] + [c[0]])
            rate = s[1] \
                + c[1] * math.log(beta) \
                + math.log(self.smoother.smooth(self.counter, sentence[-self.counter.window_size:]))
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
        cs = list(map(lambda x: ("".join(x[0]), x[1]), cs))
        return cs

    def _pickup_candidate_item(self, word, distance):
        return list(map(lambda w: (w, self.distance.measure(word, w)),
                    filter(lambda w: self.distance.measure(word, w) <= distance,
                    self.tokenizer.vocab)))

    def fix(self, sentence):
        pass
