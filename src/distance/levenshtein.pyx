import numpy as np


class Levenshtein(object):
    '''
    Levenshtein distance measurement module
    '''
    def get_char_type(self, c):
        return "others"

    def measure(self, word1, word2):
        cdef int l_w1 = len(word1)
        cdef int l_w2 = len(word2)
        cdef int i, j
        cdef int[:, :] distances = np.zeros((l_w1 + 1, l_w2 + 1), dtype=np.int32)

        for i in range(l_w1 + 1):
            distances[i][0] = i

        for j in range(l_w2 + 1):
            distances[0][j] = j

        for i in range(1, l_w1 + 1):
            for j in range(1, l_w2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    x = 0
                else:
                    if self.get_char_type(word1[i - 1]) != self.get_char_type(word2[j - 1]):
                        x = 2
                    else:
                        x = 1

                distances[i][j] = min(
                    distances[i - 1][j] + 1,
                    distances[i][j - 1] + 1,
                    distances[i - 1][j - 1] + x)
        return distances[l_w1][l_w2]
