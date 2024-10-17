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
