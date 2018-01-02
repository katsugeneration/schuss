import sentencepiece as spm


class SentencePieceTokenizer(object):
    '''
    SentencePiece Tokenizer
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
        tokenize sentence using SentencePiece
        '''
        return self.sp.EncodeAsPieces(sentence)
