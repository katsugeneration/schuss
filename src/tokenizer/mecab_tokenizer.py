# -*- coding: utf-8 -*-
import subprocess


class MecabTokenizer(object):
    '''
    Mecab Tokenizer
    '''
    def __init__(self):
        self.vocab = []

    def encode(self, sentence):
        sentence = sentence.replace("\'", "\"")
        f = subprocess.check_output("echo \'" + sentence + "\' | mecab -Owakati", shell=True).decode("utf-8", "ignore")

        return f.splitlines()[0].strip().split()
