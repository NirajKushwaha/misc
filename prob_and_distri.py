#### Contains functions for operations related to probabilities and distributions.

from .utils import *

def empirical_ccdf(samples, plot=False, return_data=False):
    """
    Returns ccdf of observed data.
    Only works for discrete data.
    
    Parameters
    ----------
    samples : ndarray
        1D array of observed data.
    plot : bool, optional
        If True, plot the ccdf.
    return_data : bool, optional
        If True, return the ccdf data.
    
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

    if(plot):
        plt.loglog(dt)
    if(return_data):
        return dt

def empirical_pdf(samples):
    """
    Returns pdf of observed data.
    Only works for discrete data.

    Parameters
    ----------
    dt : ndarray

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

    dt = dt.drop_duplicates()

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

def joint_probability(*data_series):
    """
    Calculate the joint probability distribution of multiple discrete data series.

    Parameters
    ----------
    *data_series
        Multiple data series as input, each represented as a list or iterable.

    Returns
    -------
    dict
        A dictionary where keys are tuples representing combinations of values
        from the input series, and values are the corresponding joint probabilities.
    """

    if len(data_series) < 2:
        raise ValueError("At least two data series are required for joint probability calculation.")

    series_lengths = set(len(series) for series in data_series)
    if len(series_lengths) != 1:
        raise ValueError("All data series must have the same length.")

    num_data_points = len(data_series[0])

    joint_probabilities = Counter()

    for data_points in zip(*data_series):
        joint_probabilities[data_points] += 1

    for key in joint_probabilities:
        joint_probabilities[key] /= num_data_points

    return joint_probabilities

def Freedman_Diaconis_bins(data, return_bin_width=False):
    """
    Calculate the number of bins for a histogram using the Freedman-Diaconis rule.

    Parameters
    ----------
    data : array-like
        The data to be used for the histogram.
    return_bin_width : bool, optional
        If True, return both the number of bins and the bin width.
    
    Returns
    -------
    int
        The number of bins to use for the histogram
    """

    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * IQR * len(data) ** (-1/3)
    num_bins = int((np.max(data) - np.min(data)) / bin_width)

    if(return_bin_width):
        return num_bins, bin_width
    else:
        return num_bins
