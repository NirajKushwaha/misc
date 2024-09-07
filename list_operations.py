from utils import *

def normalize_to_range(lst, x, y):
    """
    Normalize a list of values to a given range.

    Parameters
    ----------
    lst : list like
    x : float
    y : float

    Return
    ------
    list
    """

    min_val = min(lst)
    max_val = max(lst)
    
    # Edge case: if all values are the same, return a list of `x`
    if min_val == max_val:
        return [x] * len(lst)

    return [(y - x) * (val - min_val) / (max_val - min_val) + x for val in lst]