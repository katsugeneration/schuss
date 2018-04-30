# -*- coding: utf-8 -*-
import math
import subprocess
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

        self.tokenizer.vocab = list(set(self.tokenizer.vocab) | set(self.counter.vocab)
                                    | set([chr(c) for c in range(ord("\u3041"), ord("\u3096"))])
                                    | set([chr(c) for c in range(ord("\u30A1"), ord("\u30F6"))]))

    def detect(self, sentence, correct_threshold=0.001):
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
            c, s, next = args
            sentence = (s[0] + [c[0]])
            rate = s[1] \
                + c[1] * math.log(beta) \
                + math.log(self.smoother.smooth(self.counter, sentence[-self.counter.window_size:])) \
                + math.log(self.smoother.smooth(self.counter, sentence[-self.counter.window_size+1:] + [next]))
            return (sentence, rate)

        for i, candidate in enumerate(candidates):
            before_cs = cs
            cs = []
            next = words[i+1] if i+1 < len(words) else ""

            def items():
                for c in candidate:
                    for s in before_cs:
                        yield (c, s, next)

            cs = map(_calc, items())
            cs = sorted(cs, key=lambda r: r[1], reverse=True)[:num * 10]
            if len(cs) > num:
                cs = self._select_candidate(cs, words[i+1:], num)
        cs = list(map(lambda x: ("".join(x[0]), x[1]), cs))
        return cs

    def _pickup_candidate_item(self, word, distance):
        return list(map(lambda w: (w, self.distance.measure(word, w)),
                    filter(lambda w: abs(len(word) - len(w)) <= distance and self.distance.measure(word, w) <= distance,
                    self.tokenizer.vocab)))

    def _select_candidate(self, sentences, words, num):
        def get_value(sentence):
            s = "".join(sentence[0] + words).replace("\'", "\"")
            f = subprocess.check_output("echo \'" + s + "\' | mecab -F\"%pc \" -E\"EOS\"", shell=True).decode("utf-8", "ignore")
            return (sentence[0], sentence[1], int(f.split()[-2]))

        sentences = map(get_value, sentences)
        sentences = list(map(lambda x: (x[0], x[1]), sorted(sentences, key=lambda r: r[2])[:num]))
        return sentences

    def fix(self, sentence):
        pass
