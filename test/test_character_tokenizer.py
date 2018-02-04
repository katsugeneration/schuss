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
        from character_tokenizer import CharacterTokenizer
        tokenizer = CharacterTokenizer()
        assert_equal([], tokenizer.vocab)

    def test_encode(self):
        from character_tokenizer import CharacterTokenizer
        tokenizer = CharacterTokenizer()
        words = tokenizer.encode("今日は晴れです")
        assert_equal(list("今日は晴れです"), words)
