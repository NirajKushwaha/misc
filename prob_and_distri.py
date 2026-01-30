#### Contains functions for operations related to probabilities and distributions.

from .utils import *

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import find_peaks

def empirical_ccdf(samples, ax=None, plot=True, return_data=False, plot_label=""):
    """
    Plots ccdf of observed data.
    Only works for discrete data.
    This function is to be used for a very basic first look at the data.
    
    Parameters
    ----------
    samples : ndarray
        1D array of observed data.
    ax : matplotlib.axes.Axes, None
        Axes to plot on. If None, creates a new figure and axes.
    plot : bool, True
        If True, plot the ccdf.
    return_data : bool, False
        If True, return the ccdf data.
    plot_label : str, ""
        Label for the plot.
    
    Returns
    -------
    pd.Series
    """

    if ax is None:
        fig, ax = plt.subplots()

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
        ax.loglog(dt, marker='.', label=plot_label)
        ax.grid(True, which="both", ls='--', alpha=0.6)
    if(return_data):
        return dt

def empirical_ccdf_continuous(samples, ax=None, plot=True, return_data=False, plot_label=""):
    """
    TEST IT MORE THOROUGHLY.

    Computes and optionally plots the empirical CCDF of continuous data
    without repeated vertical stacks (one point per unique x).
    """

    if ax is None:
        fig, ax = plt.subplots()

    x = np.asarray(samples, dtype=float)

    x_unique, counts = np.unique(x, return_counts=True)
    n = len(x)

    # Survival counts: number of samples >= x_i
    survival_counts = np.flip(np.cumsum(np.flip(counts)))
    ccdf = survival_counts / n

    # ccdf = 1-np.cumsum(counts/n)

    if plot:
        ax.loglog(x_unique, ccdf, marker='.', label=plot_label)
        # ax.set_ylabel("CCDF = P(X ≥ x)")
        ax.grid(True, which="both", ls='--', alpha=0.6)

    if return_data:
        return ccdf




def gmm_mode_analysis(
    data,
    kmax=5,
    grid_points=2000,
    peak_prominence=0.01,
    random_state=0,
):
    """
    Fit GMMs with 1..kmax components, select by BIC,
    and estimate number/location of modes in the fitted density.

    Parameters
    ----------
    data : array-like, shape (n,)
        1D samples.
    kmax : int
        Max number of Gaussian components to test.
    grid_points : int
        Resolution for evaluating density.
    peak_prominence : float
        Prominence threshold for peak detection (relative scale).
    random_state : int

    Returns
    -------
    results : dict with keys
        best_k
        bic_values
        weights
        means
        variances
        peak_locations
        peak_heights
        x_grid
        pdf
    """

    x = np.asarray(data).reshape(-1, 1)

    bics = []
    models = []

    for k in range(1, kmax + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
        )
        gmm.fit(x)
        bics.append(gmm.bic(x))
        models.append(gmm)

    best_idx = np.argmin(bics)
    best_model = models[best_idx]
    best_k = best_idx + 1

    # ----- evaluate mixture pdf on a grid -----

    xmin, xmax = np.min(x), np.max(x)
    pad = 0.1 * (xmax - xmin)
    grid = np.linspace(xmin - pad, xmax + pad, grid_points)

    pdf = np.zeros_like(grid)

    for w, mu, cov in zip(
        best_model.weights_,
        best_model.means_.flatten(),
        best_model.covariances_.flatten(),
    ):
        pdf += w * norm.pdf(grid, mu, np.sqrt(cov))

    # ----- find peaks -----

    prominence = peak_prominence * np.max(pdf)
    peak_idx, props = find_peaks(pdf, prominence=prominence)

    peak_locations = grid[peak_idx]
    peak_heights = pdf[peak_idx]

    return {
        "best_k": best_k,
        "bic_values": np.array(bics),
        "weights": best_model.weights_,
        "means": best_model.means_.flatten(),
        "variances": best_model.covariances_.flatten(),
        "peak_locations": peak_locations,
        "peak_heights": peak_heights,
        "x_grid": grid,
        "pdf": pdf,
    }

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
