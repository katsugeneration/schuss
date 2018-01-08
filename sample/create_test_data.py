import random
import argparse


j_candidate = (list(range(0x3000, 0x301C)) +
               list(range(0x3041, 0x3093)) +
               list(range(0x30A1, 0x30F6)))


def create_miss_data(line):
    pos = random.randint(0, len(line))
    line = line[:pos] + chr(random.choice(j_candidate)) + line[pos+1:]
    return line


def main():
    parser = argparse.ArgumentParser(
        description='create spell checker test data')
    parser.add_argument(
        'input', type=str,
        help='input text file for creating test data')
    parser.add_argument(
        '--rate', default=0.01, type=float,
        help='pickup lines ratio')
    parser.add_argument(
        '--correct_rate', default=0.5, type=float,
        help='test data correct data ratio')
    parser.add_argument(
        '--output', default="test.txt", type=str,
        help='test data output path')
    args = parser.parse_args()

    w = open(args.output, 'w', encoding='utf-8')
    w.write("\t".join(["data", "correct", "flag"]) + "\n")
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if random.random() > args.rate:
                continue

            if random.random() >= args.correct_rate:
                d = line
                f = "1"
            else:
                d = create_miss_data(line)
                f = "0"
            w.write("\t".join([d, line, f]) + "\n")
            
    w.close()


if __name__ == '__main__':
    main()
