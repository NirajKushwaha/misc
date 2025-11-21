#### Contains functions for operations related to probabilities and distributions.

from .utils import *

def empirical_ccdf(samples, ax=None, plot=True, return_data=False):
    """
    Plots ccdf of observed data.
    Only works for discrete data.
    This function is to be used for a very basic first look at the data.
    
    Parameters
    ----------
    samples : ndarray
        1D array of observed data.
    plot : bool, True
        If True, plot the ccdf.
    return_data : bool, False
        If True, return the ccdf data.
    
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
        ax.loglog(dt, marker='.')
        ax.grid(True, which="both", ls='--', alpha=0.6)
    if(return_data):
        return dt

def empirical_ccdf_continuous(samples, ax=None, plot=True, return_data=False):
    """
    Computes and optionally plots the empirical CCDF of continuous data
    without repeated vertical stacks (one point per unique x).
    """

    if ax is None:
        fig, ax = plt.subplots()

    x = np.asarray(samples, dtype=float)

    # Sort and get unique values + counts
    x_unique, counts = np.unique(x, return_counts=True)
    n = len(x)

    # Survival counts: number of samples >= x_i
    survival_counts = np.flip(np.cumsum(np.flip(counts)))

    # CCDF = survival_count / N
    ccdf = survival_counts / n

    ccdf_series = pd.Series(ccdf, index=x_unique)

    if plot:
        ax.loglog(ccdf_series.index, ccdf_series.values, marker='.')
        # ax.set_ylabel("CCDF = P(X ≥ x)")
        ax.grid(True, which="both", ls='--', alpha=0.6)

    if return_data:
        return ccdf_series

def plot_discrete_ccdf(
    simulations,
    ax=None,
    xlabel='Value',
    error='std',           # 'std' or 'sem'
    use_log_y=True,
    use_log_x=True,
    capsize=3,
    plot_errors=True,
    plot_label=""
):
    """
    THIS CODE HASNE'T BEEN TESTED THOROUGHLY.

    Plot discrete CCDF (P(X > x)) with error bars across simulations.
    Interpolation is NOT used; CCDF is computed exactly on integer grid.

    Behaviour:
      - If `simulations` contains a single array, plot its CCDF without error bars.
      - If multiple simulations are provided, plot the mean CCDF and error bars
        (std or sem). If the computed error is all NaN, fall back to plotting the mean only.

    Parameters
    ----------
    simulations : list of 1D numpy arrays (integer-valued)
    ax : matplotlib.axes.Axes or None
    xlabel : label for x axis
    error : 'std' or 'sem' for whether to plot standard deviation or standard error
    use_log_y : if True, set yscale to 'log' (mask zeros)
    use_log_x : if True, set xscale to 'log' (requires x>0)
    capsize : errorbar capsize
    """
    if ax is None:
        fig, ax = plt.subplots()

    if len(simulations) == 0:
        raise ValueError("simulations list is empty")

    # Ensure integer arrays
    sims = [np.asarray(s).astype(int) for s in simulations]
    # Global support
    global_min = min(s.min() for s in sims if s.size > 0)
    global_max = max(s.max() for s in sims if s.size > 0)
    xs = np.arange(global_min, global_max + 1)

    # Preallocate ccdf matrix: shape (n_sims, len(xs))
    n_sims = len(sims)
    ccdfs = np.full((n_sims, xs.size), np.nan, dtype=float)

    for i, s in enumerate(sims):
        if s.size == 0:
            # if a simulation has no observations, leave row as NaNs
            continue

        # bincount aligned to global_min..global_max
        offset = global_min
        counts = np.bincount(s - offset, minlength=xs.size)
        N = counts.sum()
        if N == 0:
            continue
        # empirical CDF at each grid value v: P(X <= v)
        cdf = np.cumsum(counts) / N
        # CCDF as P(X > x) = 1 - P(X <= x)
        ccdf = 1.0 - cdf
        ccdfs[i, :] = ccdf

    # If there is only one simulation (or effectively one non-empty),
    # prefer plotting that single CCDF without error bars.
    nonempty_count_per_sim = np.array([0 if s.size == 0 else 1 for s in sims])
    # Compute mean and errors in the usual case (nan-aware)
    mean_ccdf = np.nanmean(ccdfs, axis=0)

    if n_sims == 1:
        err = None
    else:
        if error == 'std':
            err = np.nanstd(ccdfs, axis=0)
        elif error == 'sem':
            # sem uses n-1 by default; handle variable counts
            n_nonan = np.sum(~np.isnan(ccdfs), axis=0)
            # ddof=1 for sample std; result will be nan when n_nonan<=1
            std = np.nanstd(ccdfs, axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                err = std / np.sqrt(n_nonan)
                err[n_nonan <= 1] = np.nan
        else:
            raise ValueError("error must be 'std' or 'sem'")

        # If err is entirely NaN (e.g., every column had n_nonan<=1), don't plot errorbars.
        if np.all(np.isnan(err)):
            err = None

    # For plotting on log y: remove points where mean_ccdf == 0 or NaN
    plot_mask = ~np.isnan(mean_ccdf)
    if use_log_y:
        plot_mask &= (mean_ccdf > 0)

    # For log-x: ensure xs>0
    if use_log_x:
        if np.any(xs <= 0):
            # cannot set log-x when xs contains nonpositive values;
            # drop nonpositive xs from plotting (but still compute for them)
            plot_mask &= (xs > 0)

    if not np.any(plot_mask):
        raise RuntimeError("No valid points to plot (all CCDFs are zero/NaN given log options).")

    # Decide whether to plot with error bars or not
    if err is None:
        # single simulation or no valid error -> plot mean only
        ax.plot(xs[plot_mask], mean_ccdf[plot_mask], '.-', label=f'{plot_label}' if n_sims == 1 else 'Mean CCDF')
    else:
        if plot_errors:
            ax.errorbar(
                xs[plot_mask],
                mean_ccdf[plot_mask],
                yerr=err[plot_mask],
                fmt='.-',
                capsize=capsize,
                label=f'{plot_label}'
            )
        else:
            ax.plot(xs[plot_mask], mean_ccdf[plot_mask], '.-', label=f'{plot_label}')

    if use_log_y:
        ax.set_yscale('log')
    if use_log_x:
        ax.set_xscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('CCDF')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend()
    return ax

def plot_continuous_ccdf(
    simulations,
    ax=None,
    xlabel='Value',
    error='std',           # 'std' or 'sem'
    use_log_y=True,
    use_log_x=True,
    capsize=3,
    plot_errors=True,
    plot_label="",
    xs=None,                # optional evaluation points (1D array). If None, auto generates.
    num=200,                # when xs is None, number of points to evaluate on
    percentile_range=(0.0, 100.0),  # trim extremes when auto-generating xs (useful for heavy tails)
    step=True               # plot as steps (empirical CCDF) if True, else connect points
):
    """
    THIS CODE HASNE'T BEEN TESTED THOROUGHLY.

    Plot empirical CCDF (P(X > x)) for continuous-valued simulations.

    Behavior:
      - If `simulations` contains a single array, plot its empirical CCDF without error bars.
      - If multiple simulations are provided, compute CCDF at a common grid `xs` for each
        simulation and plot mean CCDF with error bands (std or sem). If the computed error
        is all NaN, fall back to plotting the mean only.

    Parameters
    ----------
    simulations : list of 1D numpy arrays (continuous-valued)
    ax : matplotlib.axes.Axes or None
    xlabel : str
    error : 'std' or 'sem'
    use_log_y, use_log_x : bool
    capsize : int
    plot_errors : bool
    plot_label : str
    xs : None or 1D array of evaluation points. If None, will be auto-generated.
    num : int, number of evaluation points when xs is None
    percentile_range : (low_pct, high_pct) to trim extremes when auto-generating xs
    step : bool, whether to use a step-style plot (recommended for ECDF/CCDF)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # validate input list
    if len(simulations) == 0:
        raise ValueError("simulations list is empty")

    # convert to 1D numpy arrays
    sims = [np.asarray(s).ravel() for s in simulations]
    n_sims = len(sims)

    # find non-empty sims
    nonempty_idxs = [i for i,s in enumerate(sims) if s.size > 0]
    if len(nonempty_idxs) == 0:
        raise ValueError("All provided simulations are empty.")

    # determine evaluation grid xs
    if xs is None:
        # compute global percentiles to avoid extreme outliers dominating the grid
        low_pct, high_pct = percentile_range
        # gather all values from non-empty sims into a single concatenated array (may be large)
        all_vals = np.concatenate([s for s in sims if s.size > 0])
        vmin = np.percentile(all_vals, low_pct)
        vmax = np.percentile(all_vals, high_pct)
        if vmin == vmax:
            # degenerate: expand small epsilon
            eps = np.abs(vmin) * 1e-6 if vmin != 0 else 1e-6
            vmin -= eps
            vmax += eps
        xs = np.linspace(vmin, vmax, num)
    else:
        xs = np.asarray(xs).ravel()
        if xs.size == 0:
            raise ValueError("Provided xs is empty.")

    m = xs.size

    # Preallocate: shape (n_sims, m)
    ccdfs = np.full((n_sims, m), np.nan, dtype=float)

    # For each simulation, compute empirical CCDF at xs using sorted array + searchsorted
    # CCDF(x) = P(X > x) = (number of samples strictly greater than x) / N
    for i, s in enumerate(sims):
        if s.size == 0:
            continue
        sorted_s = np.sort(s)
        N = sorted_s.size
        # use side='right' to count values <= x, so N - idx => #greater-than
        idx = np.searchsorted(sorted_s, xs, side='right')
        ccdf_vals = (N - idx) / float(N)
        ccdfs[i, :] = ccdf_vals

    # compute mean and error across sims (nan-aware)
    mean_ccdf = np.nanmean(ccdfs, axis=0)

    # determine error vector
    if n_sims == 1:
        err = None
    else:
        if error == 'std':
            err = np.nanstd(ccdfs, axis=0, ddof=0)
        elif error == 'sem':
            # use sample std (ddof=1) and divide by sqrt(n_nonan)
            n_nonan = np.sum(~np.isnan(ccdfs), axis=0)
            std = np.nanstd(ccdfs, axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                err = std / np.sqrt(n_nonan)
            # where n_nonan <= 1 sem is meaningless -> set nan
            err[n_nonan <= 1] = np.nan
        else:
            raise ValueError("error must be 'std' or 'sem'")

        if np.all(np.isnan(err)):
            err = None

    # apply masks for plotting with log scales
    plot_mask = ~np.isnan(mean_ccdf)
    if use_log_y:
        plot_mask &= (mean_ccdf > 0)
    if use_log_x:
        plot_mask &= (xs > 0)

    if not np.any(plot_mask):
        raise RuntimeError("No valid points to plot (all CCDFs are zero/NaN given log options).")

    xs_plot = xs[plot_mask]
    mean_plot = mean_ccdf[plot_mask]
    err_plot = None if err is None else err[plot_mask]

    # plotting: step style recommended to reflect empirical CCDF shape
    drawstyle = 'steps-post' if step else None

    label = plot_label if plot_label else ('Mean CCDF' if n_sims > 1 else 'CCDF')

    if err is None or (not plot_errors):
        # just plot mean
        if step:
            ax.step(xs_plot, mean_plot, where='post', label=label)
        else:
            if drawstyle:
                ax.plot(xs_plot, mean_plot, '.-', label=label, drawstyle=drawstyle)
            else:
                ax.plot(xs_plot, mean_plot, '.-', label=label)
    else:
        # use errorbars (vertical)
        # errorbar with drawstyle isn't supported; we plot points+errorbars and optionally a step line behind
        ax.errorbar(xs_plot, mean_plot, yerr=err_plot, fmt='.', capsize=capsize, label=label)
        if step:
            # draw step line for the mean
            ax.step(xs_plot, mean_plot, where='post', alpha=0.7)
        else:
            ax.plot(xs_plot, mean_plot, '.-', alpha=0.7)

    if use_log_y:
        ax.set_yscale('log')
    if use_log_x:
        ax.set_xscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('CCDF')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend()
    return ax

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
