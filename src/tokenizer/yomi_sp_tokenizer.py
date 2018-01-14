import subprocess
import sentencepiece as spm


class YomiSPTokenizer(object):
    '''
    Japanese Yomi SentencePiece Tokenizer
    '''
    def __init__(self, model_path, vocab_path):
        '''
        :parameters:
            model_path: SentencePiece .model path
            vocab_path: SentencePiece .vocab path
        '''
        self.sp = spm.SentencePieceProcessor()
        assert self.sp.Load(model_path)

        self.vocab = []
        with open(vocab_path) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                k, v = line.split()[:2]
                self.vocab.append(k)

    def encode(self, sentence):
        '''
        tokenize yomi sentence using SentencePiece
        '''
        sentence = sentence.replace("\'", "\"")
        f = subprocess.check_output("echo \'" + sentence + "\' | mecab -Oyomi", shell=True).decode("utf-8", "ignore")
        return self.sp.EncodeAsPieces(f.splitlines()[0].strip())
