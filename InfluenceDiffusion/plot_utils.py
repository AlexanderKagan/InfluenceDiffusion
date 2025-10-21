import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_with_conf_intervals", "plot_hist_with_normal_fit"]


def plot_with_conf_intervals(x_true, x_pred, conf_intervals=None,
                             fontsize=12, figsize=(10, 8), color="blue", ax=None,
                             xlab="True activation probability",
                             ylab="Predicted activation probability"):
    """
    Plot a scatter plot of `x_true` vs `x_pred` with confidence intervals as a filled area
    and a diagonal line y=x.

    Parameters
    ----------
    x_true : array-like
        True values.
    x_pred : array-like
        Predicted values.
    conf_intervals : array-like, shape (2, len(x_pred)), optional
        Lower and upper bounds of the confidence intervals. If None, no intervals are plotted.
    fontsize : int, default 12
        Font size for axis labels.
    figsize : tuple, default (10, 8)
        Figure size in inches (width, height).
    color : str or tuple, default "blue"
        Color of the scatter points and confidence interval area.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes are created.
    xlab : str, default "True activation probability"
        Label for the x-axis.
    ylab : str, default "Predicted activation probability"
        Label for the y-axis.
    """
    assert len(x_pred) == len(x_true), "x_pred and x_true must have the same length"
    if conf_intervals is not None:
        assert conf_intervals.shape[1] == len(x_pred), "conf_intervals must have same second dim as x_pred"
        assert conf_intervals.shape[0] == 2, "conf_intervals must have two rows for lower and upper bounds"

    # Sort by x_true so fill_between works correctly
    sort_idx = np.argsort(x_true)
    x_true_sorted = x_true[sort_idx]
    x_pred_sorted = x_pred[sort_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x_true_sorted, x_pred_sorted, color=color)

    # Filled confidence intervals
    if conf_intervals is not None:
        conf_lower_sorted = conf_intervals[0, sort_idx]
        conf_upper_sorted = conf_intervals[1, sort_idx]
        ax.fill_between(x_true_sorted, conf_lower_sorted, conf_upper_sorted,
                        color=color, alpha=0.2)

    # Diagonal line y=x
    x_concat = np.hstack([x_true, x_pred])
    min_x, max_x = np.min(x_concat), np.max(x_concat)
    ax.plot([min_x, max_x], [min_x, max_x], linestyle='--', color="black")

    # Labels
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)


def plot_hist_with_normal_fit(sample, true_value, true_std=None, n_bins=20):
    """
    Plot a histogram of a sample with a fitted normal curve and a vertical line at the true value.

    Parameters
    ----------
    sample : array-like
        Sample data points.
    true_value : float
        The true value.
    true_std : float, optional
        The true standard deviation, used to plot the theoretical normal curve.
    n_bins : int, default 20
        Number of bins for the histogram.
    """
    from scipy.stats import norm

    # Plot the histogram
    plt.hist(sample, bins=n_bins, density=True, alpha=0.6, color='g', edgecolor='black')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mean, std = norm.fit(sample)

    # Plot the fitted normal curve
    p_fit = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, p_fit, 'b', linewidth=2, label="Fitted Gaussian")

    # Plot the theoretical normal curve
    if true_std is not None:
        p = norm.pdf(x, loc=true_value, scale=true_std)
        plt.plot(x, p, 'r', linewidth=2, label="Theoretical Gaussian")

    # Plot vertical line at the true value
    plt.axvline(true_value, color='black', linestyle='--', linewidth=1.5, label="True value")

    # Labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
