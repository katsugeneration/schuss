from parameterized import parameterized
from nose.tools import assert_equal, assert_true
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src")
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir(dir_path)


class TestSPTokenizer:
    def test_init(self):
        from distance.levenshtein import Levenshtein
        Levenshtein()

    def test_measure(self):
        from distance.levenshtein import Levenshtein
        l = Levenshtein()
        assert_equal(0, l.measure("あいう", "あいう"))
        assert_equal(1, l.measure("あいう", "あい"))
        assert_equal(1, l.measure("いう", "あいう"))
        assert_equal(1, l.measure("あいう", "あえう"))
        assert_equal(2, l.measure("あいう", "あ"))
        assert_equal(2, l.measure("あいう", "あいうえお"))
        assert_equal(2, l.measure("あいう", "あえいうお"))
        assert_equal(2, l.measure("あ", "愛"))
        assert_equal(2, l.measure("愛", "「"))
        assert_equal(1, l.measure("あ", "「"))
