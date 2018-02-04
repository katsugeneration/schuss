# -*- coding: utf-8 -*-
import subprocess


class CharacterTokenizer(object):
    '''
    Japanese Yomi Tokenizer
    '''
    def __init__(self):
        self.vocab = []

    def encode(self, sentence):
        return list(sentence)
