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

        schuss = Schuss(lc, tokenizer, smoother, None)
        words, counts = schuss.detect("北にアゼルバイジャン、アルメニア、トルクメニスタン。東にパキスタン、アフガニスタン、西にトルコ、イラクと境を接する", correct_threshold=correct_threshold)
        arr = ['、', 'トルクメニスタン', '。', '東', 'に']
        for i, s in enumerate(sliding_window(arr, window_size)):
            if smoother.smooth(lc, s) < correct_threshold:
                assert_true(counts[i + 5] < 0)
                assert_true(counts[i + 6] < 0)
                assert_true(counts[i + 7] < 0)

        words, counts = schuss.detect("北にアゼルバイジャン、アルメニア、トルクメニスタン、東にパキスタン、アフガニスタン、西にトルコ、イラクと境を接する", correct_threshold=correct_threshold)
        arr = ['、', 'トルクメニスタン', '、', '東', 'に']
        for i, s in enumerate(sliding_window(arr, window_size)):
            if smoother.smooth(lc, s) < 0.01:
                assert_true(counts[i + 5] < 0)
                assert_true(counts[i + 6] < 0)
                assert_true(counts[i + 7] < 0)

    def test_pickup_item(self):
        from counter.lossy_counting import LossyCountingNGram
        from tokenizer.sp_tokenizer import SentencePieceTokenizer
        from smoother.laplace_smoother import LaplaceSmoother
        from distance.levenshtein import Levenshtein
        from schuss import Schuss

        lc = LossyCountingNGram(window_size=3, epsilon=1e-5)
        tokenizer = SentencePieceTokenizer('data/test.model', 'data/test.vocab')
        smoother = LaplaceSmoother(delta=1)
        l = Levenshtein()

        schuss = Schuss(lc, tokenizer, smoother, l)
        assert_equal([('トルクメニスタン', 1)], schuss._pickup_candidate_item("トルクメニステン", 2))
        assert_true([] != schuss._pickup_candidate_item("であろ", 2))
        assert_equal([], schuss._pickup_candidate_item("ああああ", 2))
        assert_equal([], schuss._pickup_candidate_item("トルクメニステ", 1))
        assert_true([] != schuss._pickup_candidate_item("ああああ", 3))

    @parameterized([
        (3, 1e-4, 0.01)
    ])
    def test_pickup(self, window_size, epsilon, correct_threshold):
        from counter.lossy_counting import LossyCountingNGram
        from tokenizer.sp_tokenizer import SentencePieceTokenizer
        from smoother.laplace_smoother import LaplaceSmoother
        from distance.levenshtein import Levenshtein
        from schuss import Schuss

        lc = LossyCountingNGram(window_size=window_size, epsilon=epsilon)
        tokenizer = SentencePieceTokenizer('data/test.model', 'data/test.vocab')
        smoother = LaplaceSmoother(delta=1)
        l = Levenshtein()
        with open('data/sp.txt', 'r', encoding='utf-8') as f:
            lc.fit(f)

        schuss = Schuss(lc, tokenizer, smoother, l)
        # words, counts = schuss.detect("北にアゼルバイジャン、アルメニア、トルクメニスタン。東にパキスタン、アフガニスタン、西にトルコ、イラクと境を接する", correct_threshold=correct_threshold)
        words, counts = schuss.detect("今日は晴れますか？", correct_threshold=correct_threshold)
        ret = schuss.pickup(words, counts)
        print(ret)
