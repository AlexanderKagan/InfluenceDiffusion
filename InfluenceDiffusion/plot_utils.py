import numpy as np
import matplotlib.pyplot as plt


def plot_with_conf_intervals(x_true, x_pred, conf_intervals=None,
                             fontsize=12, figsize=(10, 8), color="blue", ax=None,
                             xlab="True activation probability",
                             ylab="Predicted activation probability"):
    """
    Plot a scatter plot of `x_true` vs `x_pred` with confidence intervals as a filled area
    and a diagonal line y=x.

    :param x_true: Array-like, true values.
    :param x_pred: Array-like, predicted values.
    :param conf_intervals: 2D array of shape (2, len(x_pred)),
                           containing the lower and upper bounds of the confidence intervals.
                           If None, no confidence intervals are plotted.
    :param fontsize: int, font size for axis labels. Default is 12.
    :param figsize: tuple, size of the figure (width, height) in inches. Default is (10, 8).
    :param color: str or tuple, color of the scatter points and confidence interval area. Default is "blue".
    :param ax: matplotlib.axes.Axes, optional existing axes to plot on.
               If None, a new figure and axes are created.
    :param xlab: str, label for the x-axis. Default is "True activation probability".
    :param ylab: str, label for the y-axis. Default is "Predicted activation probability".
    """
    assert len(x_pred) == len(x_true), "x_pred and x_true must have the same length"
    assert conf_intervals.shape[1] == len(x_pred), "conf_intervals must have the same second dim as x_pred"
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
    min_x, max_x = np.min(x_true), np.max(x_true)
    min_y, max_y = np.min(x_pred), np.max(x_pred)
    ax.plot([min_x, max_x], [min_y, max_y], linestyle='--', color="black")

    # Labels
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)


def plot_hist_with_normal_fit(sample, true_value, true_std=None, n_bins=20):
    """
    Plot a histogram of a sample with a fitted normal curve and a vertical line at the true value.

    :param sample: Array-like, sample data points
    :param true_value: Float, the true value
    :param true_std: Float, the true std
    :param n_bins: Int, the number of histogram bins
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

    # Create the normal distribution's PDF
    if true_std is not None:
        p = norm.pdf(x, loc=true_value, scale=true_std)
        plt.plot(x, p, 'r', linewidth=2, label="Theoretical Gaussian")

    # Plot the vertical line at the true value
    plt.axvline(true_value, color='black', linestyle='--', linewidth=1.5, label="True value")

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
