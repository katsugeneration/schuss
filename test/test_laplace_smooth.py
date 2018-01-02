from parameterized import parameterized
from nose.tools import assert_equal, assert_true
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src/smoother")
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.join(dir_path, "../src/counter")
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir(dir_path)


class TestSPTokenizer:
    def test_init(self):
        from laplace_smoother import LaplaceSmoother
        LaplaceSmoother(delta=1)

    @parameterized([(1, ), (2, )])
    def test_smooth(self, delta):
        from laplace_smoother import LaplaceSmoother
        from lossy_counting import LossyCountingNGram
        lc = LossyCountingNGram(window_size=2, epsilon=1e-2)
        smoother = LaplaceSmoother(delta=delta)
        with open('data/wakati.txt', 'r', encoding='utf-8') as f:
            lc.fit(f)

        ret = lc.search(["として"])
        assert_equal((4 + delta) / (sum(ret.values()) + len(ret) * delta), smoother.smooth(lc, ["として", "使用"]))
        assert_equal((delta) / (sum(lc._items.values()) + len(lc._items) * delta), smoother.smooth(lc, ["と", "は"]))
