### Contains functions for operation for list-like objects.

from .utils import *

def normalize_to_range(lst, x, y, min_val=None, max_val=None):
    """
    Normalize a list of values to a given range, ignoring np.nan values.

    Parameters
    ----------
    lst : list-like
    x : float
        Minimum value in the normalized list.
    y : float
        Maximum value in the normalized list.
    min_val : float, optional
        Minimum cutoff value to use. If None, the minimum value in the list is used. Default is None.
    max_val : float, optional
        Maximum cutoff value to use. If None, the maximum value in the list is used. Default is None.

    Returns
    -------
    ndarray
        A ndarray of normalized values within the range [x, y], with np.nan values unchanged.
    """

    arr = np.array(lst, dtype=np.float64)

    if(min_val is None):
        min_val = np.nanmin(arr)
    if(max_val is None):
        max_val = np.nanmax(arr)

    # Edge case: if all non-nan values are the same, return a list of `x`, preserving np.nan values
    if min_val == max_val:
        return [x if not np.isnan(val) else np.nan for val in arr]

    normalized = [(y - x) * (val - min_val) / (max_val - min_val) + x if not np.isnan(val) else np.nan for val in arr]

    return np.array(normalized)

def min_in_parallel(array):
    """
    Find the minimum of a 1D numpy array using parallelization.
    
    Parameters
    ----------
    array : 1D ndarray
    
    Returns
    -------
    float
    """

    try:
        import multiprocessing
    except ImportError:
        raise ImportError("'multiprocessing' package not found. Please install it using either pip or conda.")
    
    def find_min_in_subarray(subarray, result, idx):
        min_val = np.min(subarray)
        result[idx] = min_val


    num_processes = int(cpu_count()/2) # Adjust the number of processes as needed
    chunk_size = len(array) // num_processes

    min_values = multiprocessing.Array('d', num_processes)

    processes = []

    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else len(array)
        subarray = array[start_idx:end_idx]

        p = multiprocessing.Process(target=find_min_in_subarray, args=(subarray, min_values, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    overall_min = np.min(min_values)
    
    return overall_min

def max_in_parallel(array):
    """
    Find the maximum of a 1D numpy array using parallelization.
    
    Parameters
    ----------
    array : 1D ndarray
    
    Returns
    -------
    float
    """

    try:
        import multiprocessing
    except ImportError:
        raise ImportError("'multiprocessing' package not found. Please install it using either pip or conda.")
    
    def find_max_in_subarray(subarray, result, idx):
        max_val = np.max(subarray)
        result[idx] = max_val


    num_processes = int(cpu_count()/2) # Adjust the number of processes as needed
    chunk_size = len(array) // num_processes

    max_values = multiprocessing.Array('d', num_processes)

    processes = []

    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else len(array)
        subarray = array[start_idx:end_idx]

        p = multiprocessing.Process(target=find_max_in_subarray, args=(subarray, max_values, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    overall_max = np.max(max_values)
    
    return overall_max

def dict_arr(dictionary, list_output=False):
    """
    Returns keys and values of a dict as ndarrays.

    Parameters
    ----------
    dictionary : dict
    list_output : bool, False
        If True, returns lists instead of ndarrays.

    Returns
    -------
    ndarray, ndarray
    """

    if(list_output):
        return list(dictionary.keys()), list(dictionary.values())
    else:
        return np.array(list(dictionary.keys()), dtype=object), np.array(list(dictionary.values()), dtype=object)

def unique_tuple_int_assigner(tuples_list):
    """
    Replace each unique tuple in a list with an unique integer identifier.

    Parameters
    ----------
    tuples_list : list
    
    Returns
    -------
    list
    """

    unique_map = {}
    counter = 0

    replaced_list = []

    for tup in tuples_list:
        if tup not in unique_map:
            unique_map[tup] = counter
            counter += 1
        replaced_list.append(unique_map[tup])
        
    return replaced_list

def overlap_range(ranges):
    """Find the overlapping range from a list of (low, high) ranges.

    Parameters
    ----------
    ranges : list of tuple
        A list of (low, high) tuples representing ranges.
    
    Returns
    -------
    tuple
        A tuple (overlap_low, overlap_high) representing the overlapping range. If there is no overlap, returns (np.nan, np.nan).
    """

    lows = []
    highs = []
    for i, r in enumerate(ranges):
        if len(r) != 2:
            raise ValueError(f"Range at index {i} does not have exactly 2 elements: {r!r}")
        low, high = r
        if low > high:
            low, high = high, low
        lows.append(low)
        highs.append(high)
    overlap_low = max(lows)
    overlap_high = min(highs)
    if overlap_low <= overlap_high:
        return (overlap_low, overlap_high)
    return (np.nan, np.nan)
