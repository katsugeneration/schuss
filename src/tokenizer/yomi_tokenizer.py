# -*- coding: utf-8 -*-
import subprocess


class YomiTokenizer(object):
    '''
    Japanese Yomi Tokenizer
    '''
    def __init__(self):
        self.vocab = []

    def encode(self, sentence):
        sentence = sentence.replace("\'", "\"")
        f = subprocess.check_output("echo \'" + sentence + "\' | mecab -Oyomi", shell=True).decode("utf-8", "ignore")

        return list(f.splitlines()[0].strip())
