from parameterized import parameterized
from nose.tools import assert_equal, assert_true
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src/tokenizer")
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir(dir_path)


class TestSPTokenizer:
    def test_init(self):
        from yomi_sp_tokenizer import YomiSPTokenizer
        tokenizer = YomiSPTokenizer('data/yomi.model', 'data/yomi.vocab')
        assert_true(tokenizer.sp)
        assert_equal(len(tokenizer.vocab), 800)

    def test_encode(self):
        from yomi_sp_tokenizer import YomiSPTokenizer
        tokenizer = YomiSPTokenizer('data/yomi.model', 'data/yomi.vocab')
        words = tokenizer.encode("今日は晴れです")
        assert_true(all([w in tokenizer.vocab for w in words]))
