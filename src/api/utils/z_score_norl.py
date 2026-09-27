import numpy as np


def z_score_normalize(arr):

    a, b, c = arr.shape
    return (arr - np.mean(arr, axis=-1).reshape(a, b, 1)) / np.std(arr, axis=-1).reshape(a, b, 1)


if "__main__" == __name__:
    arr = np.random.random([30, 63, 1000])

    print(z_score_normalize(arr).shape)

