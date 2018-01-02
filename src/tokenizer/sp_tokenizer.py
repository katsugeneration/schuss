import sentencepiece as spm


class SentencePieceTokenizer(object):
    def __init__(self, model_path, vocab_path):
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
        return self.sp.EncodeAsPieces(sentence)
