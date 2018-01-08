import argparse
import pickle
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src")
if module_path not in sys.path:
    sys.path.append(module_path)


def main():
    parser = argparse.ArgumentParser(
        description='create n-gram count object')
    parser.add_argument(
        'counter', type=str,
        help='counter object pickle path')
    parser.add_argument(
        'model', type=str,
        help='sentencepiece .model path')
    parser.add_argument(
        'vocab', type=str,
        help='sentencepiece .vocab path')
    parser.add_argument(
        '--delta', default=1, type=int,
        help='laplace smoothing algorithme delta parameter')
    parser.add_argument(
        '--output_num', default=10, type=int,
        help='best n candidate output per word')
    parser.add_argument(
        '--distance', default=1, type=int,
        help='fixed candidate word distance')
    parser.add_argument(
        '--correct_threshold', default=0.01, type=float,
        help='judgement correct word probability')
    parser.add_argument(
        '--cost_threshold', default=-2, type=int,
        help='judgement fixed word cost')
    parser.add_argument(
        '--beta', default=0.1, type=float,
        help='word distance penalty parameter')
    args = parser.parse_args()

    from tokenizer.sp_tokenizer import SentencePieceTokenizer
    from tokenizer.mecab_tokenizer import MecabTokenizer
    from smoother.laplace_smoother import LaplaceSmoother
    from distance.levenshtein import Levenshtein
    from schuss import Schuss

    with open(args.counter, 'rb') as f:
        lc = pickle.load(f)
    # tokenizer = SentencePieceTokenizer(args.model, args.vocab)
    tokenizer = MecabTokenizer()
    smoother = LaplaceSmoother(delta=args.delta)
    l = Levenshtein()

    schuss = Schuss(lc, tokenizer, smoother, l)

    while True:
        sentence = input("input: ")
        import time
        start = time.time()
        words, counts = schuss.detect(sentence, correct_threshold=args.correct_threshold)
        ret = schuss.pickup(words, counts, num=args.output_num, distance=args.distance, cost_threshold=args.cost_threshold, beta=args.beta)
        print(ret)
        print(time.time() - start)


if __name__ == '__main__':
    main()
