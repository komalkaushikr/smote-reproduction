"""
SMOTE, implemented from scratch in NumPy.

This is a faithful translation of Algorithm SMOTE(T, N, k) from
Chawla et al. (2002), Section 4.2. Every block below is annotated with the
line numbers of the paper's pseudo-code so you can hold the paper next to the
code and see they say the same thing.

KEY IDEA:
  SMOTE is NOT a model. It is a data transformer. It takes the minority-class
  points and manufactures new, plausible minority points by walking a random
  fraction of the way from a real point toward one of its nearest minority
  neighbours:   new = A + gap * (B - A),  gap ~ Uniform(0, 1).
  The result is a *bigger, balanced* training set that a real classifier
  (decision tree, etc.) then learns from.
"""

import numpy as np


def _k_nearest_neighbours(sample, pool, k):
    """Return the indices (into `pool`) of the k nearest neighbours of `sample`,
    using Euclidean distance. We exclude the point itself (distance 0).

    This is the brute-force O(n^2) version. It is fine for the paper's small
    tabular datasets and it keeps the logic transparent. (imbalanced-learn uses
    a KD-tree for speed -- same result, faster.)
    """
    diffs = pool - sample                      # vector from sample to every point
    dists = np.sqrt((diffs ** 2).sum(axis=1))  # Euclidean distance to each point
    order = np.argsort(dists)                   # nearest first
    # order[0] is the sample itself (distance 0); take the next k.
    return order[1:k + 1]


def smote(minority_samples, N, k=5, random_state=None):
    """
    Generate synthetic minority-class samples.

    Parameters
    ----------
    minority_samples : ndarray, shape (T, n_features)
        The real minority-class feature vectors ONLY. (You pass in just the
        minority rows -- SMOTE never looks at the majority class.)
    N : int
        Amount of over-sampling as a percentage. Must be a multiple of 100 in
        this simple version. N=100 -> generate T new samples (double the
        minority). N=200 -> generate 2T new samples. N<100 handled below.
    k : int
        Number of nearest neighbours to consider (paper uses 5).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    synthetic : ndarray, shape ((N/100)*T, n_features)
        The newly invented minority samples (NOT including the originals).
    """
    rng = np.random.default_rng(random_state)
    minority_samples = np.asarray(minority_samples, dtype=float)
    T = len(minority_samples)
    numattrs = minority_samples.shape[1]

    # ---- pseudo-code lines 1-6: handle N < 100% ----
    # If N < 100, we only SMOTE a random subset of the minority points, one
    # synthetic sample each, then treat N as 100.
    if N < 100:
        T = int((N / 100) * T)
        idx = rng.permutation(len(minority_samples))[:T]     # line 3: randomize
        minority_samples = minority_samples[idx]
        N = 100

    # ---- line 7: N is now an integer number of "rounds" ----
    N = int(N / 100)   # e.g. 200% -> 2 synthetic samples per minority point

    synthetic = []     # line 12: Synthetic[][]

    # ---- lines 13-16: for each minority sample, find neighbours & populate ----
    for i in range(T):
        sample = minority_samples[i]
        nn_indices = _k_nearest_neighbours(sample, minority_samples, k)  # line 14

        # ---- Populate(N, i, nnarray): lines 17-26 ----
        for _ in range(N):                       # while N != 0
            nn = rng.integers(0, len(nn_indices))          # line 18: pick a neighbour
            neighbour = minority_samples[nn_indices[nn]]

            new_point = np.empty(numattrs)
            for attr in range(numattrs):          # line 19: for each feature
                dif = neighbour[attr] - sample[attr]       # line 20
                gap = rng.random()                          # line 21: gap ~ U(0,1)
                new_point[attr] = sample[attr] + gap * dif  # line 22
            synthetic.append(new_point)

    return np.array(synthetic)


def smote_resample(X, y, minority_label, N, k=5, random_state=None):
    """Convenience wrapper: return a full balanced (X, y) training set.

    Takes the whole training set, runs `smote` on just the minority rows, and
    stacks the synthetic points back on. This is what the experiment harness
    actually calls.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    minority_mask = (y == minority_label)
    minority_X = X[minority_mask]

    synthetic_X = smote(minority_X, N=N, k=k, random_state=random_state)
    synthetic_y = np.full(len(synthetic_X), minority_label)

    X_out = np.vstack([X, synthetic_X])
    y_out = np.concatenate([y, synthetic_y])
    return X_out, y_out
