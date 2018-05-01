import argparse
import sentencepiece as spm


class SentencePieceTokenizer(object):
    '''
    SentencePiece Tokenizer
    '''
    def __init__(self, model_path):
        '''
        :parameters:
            model_path: SentencePiece .model path
        '''
        self.sp = spm.SentencePieceProcessor()
        assert self.sp.Load(model_path)

    def encode(self, sentence):
        '''
        tokenize sentence using SentencePiece
        '''
        words = self.sp.EncodeAsIds(sentence)
        words = words[:50]
        if len(words) < 50:
            words += [8000] * (50 - len(words))
        return words


def main():
    parser = argparse.ArgumentParser(
        description='create spell checker test data')
    parser.add_argument(
        'input', type=str,
        help='input text file for creating test data')
    parser.add_argument(
        '--model', default=0.01, type=str,
        help='model path')
    parser.add_argument(
        '--output', default="test.txt", type=str,
        help='test data output path')
    args = parser.parse_args()

    sp = SentencePieceTokenizer(args.model)
    w = open(args.output, 'w')
    is_first = True
    for line in open(args.input, 'r'):
        if is_first:
            is_first = False
            continue
        data, correct, flag = line.split('\t')
        words = sp.encode(data)
        w.write(" ".join([str(w) for w in (words + [flag.strip()])]) + "\n")


if __name__ == '__main__':
    main()
