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
        'input', type=str,
        help='test data tsv path')
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
    from smoother.laplace_smoother import LaplaceSmoother
    from distance.levenshtein import Levenshtein
    from schuss import Schuss

    with open(args.counter, 'rb') as f:
        lc = pickle.load(f)
    tokenizer = SentencePieceTokenizer(args.model, args.vocab)
    smoother = LaplaceSmoother(delta=args.delta)
    l = Levenshtein()

    schuss = Schuss(lc, tokenizer, smoother, l)

    f = open(args.input, "r", encoding="utf-8")
    count = {True: {True: 0, False: 0}, False: {True: 0, False: 0}}
    results = []
    for line in f:
        if line.startswith('data'):
            continue
        sentence, correct, flag = line.strip().split("\t")
        words, counts = schuss.detect(sentence, correct_threshold=args.correct_threshold)
        ret = schuss.pickup(words, counts, num=args.output_num, distance=args.distance, cost_threshold=args.cost_threshold, beta=args.beta)
        count[flag == "1"][sentence == ret[0][0]] += 1
        results.append((sentence, ret[0][0]))

    print(count)
    print(results)


if __name__ == '__main__':
    main()
