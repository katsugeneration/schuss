import collections


def sliding_window(arr, window_size):
    for i in range(0, len(arr) - window_size + 1):
        yield arr[i:i+window_size]


def flatten(l):
    for el in l.values():
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
