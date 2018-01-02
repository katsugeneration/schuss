from parameterized import parameterized
from nose.tools import assert_equal, assert_true
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src/counter")
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir(dir_path)


class TestLossyCounting:
    def test_init(self):
        from lossy_counting import LossyCountingNGram
        lc = LossyCountingNGram(window_size=4, epsilon=1)
        assert_equal(lc.window_size, 4)
        assert_equal(lc.epsilon, 1)

    @parameterized([
        (2, 1e-1),
        (3, 1e-1),
        (4, 1e-1),
        (2, 1e-2),
        (3, 1e-2),
    ])
    def test_fit(self, window_size, epsilon):
        from lossy_counting import LossyCountingNGram
        lc = LossyCountingNGram(window_size=window_size, epsilon=epsilon)
        with open('data/wakati.txt', 'r', encoding='utf-8') as f:
            lc.fit(f)

            f.seek(0)
            symbol_num = 0
            for line in f:
                if line.strip() == "":
                    continue
                symbol_num += sum([1 for _ in line.strip().split()]) - window_size + 1

        assert_equal(symbol_num, lc._symbol_num)
        assert_true(int(symbol_num * epsilon) - 1 <= lc._buckets_num <= int(symbol_num * epsilon))

    @parameterized([
        (2, 1e-1),
        (2, 1e-2),
    ])
    def test_fit_count(self, window_size, epsilon):
        from lossy_counting import LossyCountingNGram
        lc = LossyCountingNGram(window_size=window_size, epsilon=epsilon)
        with open('data/wakati.txt', 'r', encoding='utf-8') as f:
            lc.fit(f)

            f.seek(0)
            symbol_num = 0
            for line in f:
                if line.strip() == "":
                    continue
                symbol_num += sum([1 for _ in line.strip().split()]) - window_size + 1

        assert_true(all([v >= lc._buckets_num for v in lc._items.values()]))
        assert_true(lc._items["で@は"] < 5 + int(symbol_num * epsilon))
        assert_true(lc._items["は@、"] < 5 + int(symbol_num * epsilon))
