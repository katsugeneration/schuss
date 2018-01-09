import argparse
import pickle
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src")
if module_path not in sys.path:
    sys.path.append(module_path)


def main():
    from counter.lossy_counting import LossyCountingNGram

    parser = argparse.ArgumentParser(
        description='create n-gram count object')
    parser.add_argument(
        'input', type=str,
        help='input tokenized sentence text encoding utf-8')
    parser.add_argument(
        '--window_size', default=2, type=int,
        help='n-gram window size')
    parser.add_argument(
        '--epsilon', default=1e-7, type=float,
        help='Lossy counting algorithme quality parameter')
    parser.add_argument(
        '--output', default="counter.pkl", type=str,
        help='object output path')
    args = parser.parse_args()

    lc = LossyCountingNGram(window_size=args.window_size, epsilon=args.epsilon)
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
        lc.fit(f)

    with open(args.output, 'wb') as f:
        pickle.dump(lc, f)


if __name__ == '__main__':
    main()
