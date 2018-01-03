from parameterized import parameterized
from nose.tools import assert_equal, assert_true
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src")
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir(dir_path)

from util.sliding import sliding_window

class TestLossyCounting:
    def test_init(self):
        from counter.lossy_counting import LossyCountingNGram
        from tokenizer.sp_tokenizer import SentencePieceTokenizer
        from smoother.laplace_smoother import LaplaceSmoother
        from schuss import Schuss

        lc = LossyCountingNGram(window_size=3, epsilon=1e-2)
        tokenizer = SentencePieceTokenizer('data/test.model', 'data/test.vocab')
        smoother = LaplaceSmoother(delta=1)
        Schuss(lc, tokenizer, smoother, None)

    @parameterized([
        (3, 1e-4, 0.01)
    ])
    def test_detect(self, window_size, epsilon, correct_threshold):
        from counter.lossy_counting import LossyCountingNGram
        from tokenizer.sp_tokenizer import SentencePieceTokenizer
        from smoother.laplace_smoother import LaplaceSmoother
        from schuss import Schuss

        lc = LossyCountingNGram(window_size=window_size, epsilon=epsilon)
        tokenizer = SentencePieceTokenizer('data/test.model', 'data/test.vocab')
        smoother = LaplaceSmoother(delta=1)
        with open('data/sp.txt', 'r', encoding='utf-8') as f:
            lc.fit(f)

        schuss = Schuss(lc, tokenizer, smoother, None, correct_threshold=correct_threshold)
        words, counts = schuss.detect("北にアゼルバイジャン、アルメニア、トルクメニスタン。東にパキスタン、アフガニスタン、西にトルコ、イラクと境を接する")
        arr = ['、', 'トルクメニスタン', '。', '東', 'に']
        for i, s in enumerate(sliding_window(arr, window_size)):
            if smoother.smooth(lc, s) < correct_threshold:
                assert_true(counts[i + 5] < 0)
                assert_true(counts[i + 6] < 0)
                assert_true(counts[i + 7] < 0)

        words, counts = schuss.detect("北にアゼルバイジャン、アルメニア、トルクメニスタン、東にパキスタン、アフガニスタン、西にトルコ、イラクと境を接する")
        arr = ['、', 'トルクメニスタン', '、', '東', 'に']
        for i, s in enumerate(sliding_window(arr, window_size)):
            if smoother.smooth(lc, s) < 0.01:
                assert_true(counts[i + 5] < 0)
                assert_true(counts[i + 6] < 0)
                assert_true(counts[i + 7] < 0)
