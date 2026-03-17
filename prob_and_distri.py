#### Contains functions for operations related to probabilities and distributions.

from .utils import *

from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def empirical_ccdf(
    samples,
    ax=None,
    plot=True,
    return_data=False,
    plot_label="",
    power_law_exponent=None,
    power_law_exponent_type="pmf",
    power_law_xmin=None,
    power_law_label=None,
    power_law_kwargs=None,
):
    """
    Plot the empirical CCDF of discrete observed data.

    This function is intended as a first visual check for heavy-tailed data.
    It computes the survival function P(X >= x) on the observed support only,
    which avoids artificial gaps from unobserved integer values. Zero-valued
    observations are included in the empirical CCDF, but omitted from the
    log-log plot because x = 0 cannot be displayed on a logarithmic axis.
    
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
    power_law_exponent : float, None
        If provided, overlay a power-law reference curve on the CCDF plot.
    power_law_exponent_type : {"pmf", "ccdf"}, "pmf"
        Interpretation of `power_law_exponent`. For a discrete power-law
        distribution with PMF proportional to x^{-alpha}, use "pmf". The
        corresponding CCDF reference decays approximately as x^{-(alpha - 1)}.
        Use "ccdf" if the supplied exponent already corresponds to the CCDF.
    power_law_xmin : int, None
        Positive x-value used to anchor the reference curve to the empirical
        CCDF. If None, uses the smallest positive observed value.
    power_law_label : str, None
        Label for the reference curve. If None, a default label is created.
    power_law_kwargs : dict, None
        Extra matplotlib keyword arguments passed to the reference plot.
    
    Returns
    -------
    pd.Series
    """

    if ax is None:
        fig, ax = plt.subplots()

    dt = np.asarray(samples)

    if dt.ndim != 1:
        raise ValueError("samples must be a 1D array-like object.")
    if dt.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.issubdtype(dt.dtype, np.integer):
        raise ValueError("empirical_ccdf only works for discrete integer data.")
    if np.any(dt < 0):
        raise ValueError("empirical_ccdf expects non-negative integer data.")

    x_unique, counts = np.unique(dt, return_counts=True)
    survival_counts = np.flip(np.cumsum(np.flip(counts)))
    ccdf = survival_counts / dt.size
    ccdf_data = pd.Series(ccdf, index=x_unique)

    if(plot):
        plot_mask = ccdf_data.index > 0
        if not np.any(plot_mask):
            raise ValueError("empirical_ccdf cannot plot when all observations are zero on a log x-axis.")

        x_plot = ccdf_data.index[plot_mask].to_numpy()
        y_plot = ccdf_data.values[plot_mask]

        ax.loglog(
            x_plot,
            y_plot,
            marker='.',
            linestyle='none',
            label=plot_label
        )

        if power_law_exponent is not None:
            if power_law_exponent_type not in {"pmf", "ccdf"}:
                raise ValueError("power_law_exponent_type must be either 'pmf' or 'ccdf'.")

            if power_law_exponent_type == "pmf":
                ccdf_exponent = power_law_exponent - 1
                if ccdf_exponent <= 0:
                    raise ValueError(
                        "For a PMF exponent reference, power_law_exponent must be greater than 1."
                    )
            else:
                ccdf_exponent = power_law_exponent
                if ccdf_exponent <= 0:
                    raise ValueError(
                        "For a CCDF exponent reference, power_law_exponent must be positive."
                    )

            if power_law_xmin is None:
                anchor_x = x_plot[0]
            else:
                anchor_x = power_law_xmin
                if anchor_x <= 0:
                    raise ValueError("power_law_xmin must be positive for log-log plotting.")
                if anchor_x not in ccdf_data.index:
                    raise ValueError("power_law_xmin must be an observed value in samples.")

            anchor_y = ccdf_data.loc[anchor_x]
            ref_mask = x_plot >= anchor_x
            x_ref = x_plot[ref_mask]
            y_ref = anchor_y * (x_ref / anchor_x) ** (-ccdf_exponent)

            ref_kwargs = {"linestyle": "--", "linewidth": 1.5, **(power_law_kwargs or {})}
            if power_law_label is None:
                if power_law_exponent_type == "pmf":
                    power_law_label = f"Power-law ref (PMF exponent={power_law_exponent:g})"
                else:
                    power_law_label = f"Power-law ref (CCDF exponent={power_law_exponent:g})"

            ax.loglog(x_ref, y_ref, label=power_law_label, **ref_kwargs)

        ax.grid(True, which="both", ls='--', alpha=0.6)
    if(return_data):
        return ccdf_data

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

def empirical_pdf(samples):
    """
    Returns pdf of observed data.
    Only works for discrete data.

    Parameters
    ----------
    samples : ndarray

    Returns
    -------
    pd.Series
    """

    dt = samples

    if np.any(dt.dtype != int):
        raise ValueError("empirical_pdf only works for discrete integer data.")

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

def count_distribution_peaks(
    x,
    grid_points=2000,
    prominence=0.01,
    extend_frac=0.05,
    min_peak_distance_val=0.5,
    plot_distri=False
    ):
    """
    Count ("significant") number of peaks in the distribution of data x using KDE and peak finding.

    Parameters
    ----------
    x : array-like, shape (n,)
        1D samples.
    grid_points : int
        Number of points in the grid for KDE evaluation.
    prominence : float
        Prominence threshold for peak detection.
    extend_frac : float
        Fraction to extend the range of x for KDE evaluation.
    min_peak_distance_val : float
        Minimum distance between peaks in the same units as x.
    plot_distri : bool
        If True, a very simple plot of the estimated distribution with detected peaks.

    Returns
    -------
    n_peaks : int
        Number of detected peaks.
    xs : ndarray
        Grid points where the PDF is evaluated.
    pdf : ndarray
        Estimated probability density function values at xs.
    peaks : ndarray
        Indices of the detected peaks in xs.
    """

    xmin, xmax = x.min(), x.max()

    # ---- handle degenerate case ----
    if np.allclose(xmin, xmax):
        xs = np.array([xmin])
        pdf = np.array([1.0])
        peaks = np.array([0])
        return 1, xs, pdf, peaks

    span = xmax - xmin

    # extend range so boundary peaks move inward
    xs = np.linspace(
        xmin - extend_frac * span,
        xmax + extend_frac * span,
        grid_points,
    )

    dx = xs[1] - xs[0]
    min_distance_pts = int(np.ceil(min_peak_distance_val / dx))

    kde = gaussian_kde(x)
    pdf = kde(xs)

    peaks, _ = find_peaks(pdf, 
                              prominence=prominence, 
                              distance=min_distance_pts)

    if plot_distri:
        plt.plot(xs, pdf/pdf.max(), label="Estimated PDF")
        plt.plot(xs[peaks], pdf[peaks]/pdf.max(), "x")
        plt.xlabel("x")
        plt.ylabel("PDF")
        plt.title(f"Number of peaks: {len(peaks)}")
        plt.legend()
        plt.show()
    return len(peaks), xs, pdf, peaks
