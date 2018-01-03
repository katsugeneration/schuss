def sliding_window(arr, window_size):
    for i in range(0, len(arr) - window_size + 1):
        yield arr[i:i+window_size]