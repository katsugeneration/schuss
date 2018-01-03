# -*- coding: utf-8 -*-
from util.sliding import sliding_window


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

    def detect(self, sentence, window_size=3, correct_threshold=0.05):
        '''
        detect sentence miss position.
        :parameters:
            window_size: considering window size before word.
            correct_threshold: probability threshold for correct word order judgement.
        '''
        words = self.tokenizer.encode(sentence)
        costs = [0] * len(words)

        for i, s in enumerate(sliding_window(words, window_size)):
            p = self.smoother.smooth(self.counter, s)
            if p >= correct_threshold:
                continue
            for j in range(i, i + window_size):
                costs[j] += -1

        return words, costs

    def pickup(self, words, costs):
        pass

    def fix(self, sentence):
        pass
