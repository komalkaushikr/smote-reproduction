"""Load the paper's datasets into a uniform (X, y) NumPy format.

For now just Pima. Adding Satimage/Mammography later means writing one more
loader function with the same (X, y, minority_label) return signature, and the
rest of the pipeline works unchanged -- that uniform interface is the point.
"""

import numpy as np
import pandas as pd


def load_pima(csv_path):
    """Pima Indian Diabetes.

    8 numeric features, target in the last column: 1 = diabetes (minority,
    268 samples), 0 = no diabetes (majority, 500 samples).

    Returns
    -------
    X : ndarray (768, 8)   feature matrix, all numeric
    y : ndarray (768,)     labels {0, 1}
    minority_label : int   which label is the minority class (here, 1)
    """
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=int)
    minority_label = 1
    return X, y, minority_label


def class_balance(y):
    """Return {label: count} so we can print the 'before' picture."""
    labels, counts = np.unique(y, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))
