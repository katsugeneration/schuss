

class Levenshtein(object):
    '''
    Levenshtein distance measurement module
    '''
    def measure(self, word1, word2):
        distances = []

        for i in range(len(word1) + 1):
            distances.append([0] * (len(word2) + 1))
            distances[i][0] = i

        for j in range(len(word2) + 1):
            distances[0][j] = j

        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    x = 0
                else:
                    x = 1

                distances[i][j] = min(
                    distances[i - 1][j] + 1,
                    distances[i][j - 1] + 1,
                    distances[i - 1][j - 1] + x)
        return distances[-1][-1]
