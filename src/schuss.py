# -*- coding: utf-8 -*-
from util.sliding import sliding_window


class Schuss(object):
    def __init__(self, counter, tokenizer, smoother, distance, window_size=3, correct_threshold=0.05):
        self.counter = counter
        self.tokenizer = tokenizer
        self.smoother = smoother
        self.distance = distance
        self.window_size = window_size
        self.correct_threshold = correct_threshold

    def detect(self, sentence):
        words = self.tokenizer.encode(sentence)
        costs = [0] * len(words)

        for i, s in enumerate(sliding_window(words, self.window_size)):
            p = self.smoother.smooth(self.counter, s)
            if p >= self.correct_threshold:
                continue
            for j in range(i, i + self.window_size):
                costs[j] += -1

        return words, costs

    def pickup(self, words, costs):
        pass

    def fix(self, sentence):
        pass
