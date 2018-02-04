# -*- coding: utf-8 -*-
class CharacterTokenizer(object):
    '''
    Character Tokenizer
    '''
    def __init__(self):
        self.vocab = []

    def encode(self, sentence):
        return list(sentence)
