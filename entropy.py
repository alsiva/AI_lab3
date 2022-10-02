import numpy as np


def entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log2(norm_counts)).sum()


print(entropy([1, 3, 5, 2, 3, 5, 3, 2, 1, 3, 4, 5]))
