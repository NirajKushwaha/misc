#### Contains functions for operations related to probabilities and distributions.

from utils import *

def empirical_ccdf(samples):
    """
    Returns ccdf of observed data.
    
    Parameters
    ----------
    samples : ndarray
    
    Returns
    -------
    pd.Series
    """
    
    dt = samples
    
    if not isinstance(dt, np.ndarray):
        # Convert x into a NumPy array
        dt = np.array(dt)
    
    dt = np.bincount(dt)
    dt = dt/dt.sum()             #For Normalization
    dt[dt == 0] = np.nan
    dt = pd.DataFrame(dt)
    dt = dt.cumsum(skipna=True)           #To get commulaative distribution
    dt = (1-dt)                    #To get complimentary commulative distribution
    dt = dt[0]          #ccdf_data
    
    return dt

def conditional_probability(x,y):
    """
    Calculate conditional probabilities for two data series.
    
    Parameters
    ----------
    x : 1D ndarray like
    y : 1D ndarray like
    
    Returns
    -------
    dict
    """
    
    assert len(x) == len(y)

    y_counts = Counter(y)
    total_samples = len(y)

    xy_counts = Counter(zip(x,y))

    cp_dict = {}
    for xy in xy_counts.keys():
        cp_dict[xy] = (xy_counts[xy]/total_samples) / (y_counts[xy[1]]/total_samples)

    return cp_dict
