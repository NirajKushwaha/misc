from utils import *

def normalize_to_range(lst, x, y):
    """
    Normalize a list of values to a given range, ignoring np.nan values.

    Parameters
    ----------
    lst : list-like
    x : float
    y : float

    Returns
    -------
    ndarray
        A ndarray of normalized values within the range [x, y], with np.nan values unchanged.
    """

    arr = np.array(lst, dtype=np.float64)

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    # Edge case: if all non-nan values are the same, return a list of `x`, preserving np.nan values
    if min_val == max_val:
        return [x if not np.isnan(val) else np.nan for val in arr]

    normalized = [(y - x) * (val - min_val) / (max_val - min_val) + x if not np.isnan(val) else np.nan for val in arr]

    return np.array(normalized)