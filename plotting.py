"""
Helper functions for data visualization.
"""
import math

import pandas as pd
import numpy as np
import plotnine as p9
import random

def shade_plot(X: np.array, Y: np.array, sigma: np.array, x_label: str, y_label: str,
              width: int = 5, height: int = 3):
    """_summary_

    Parameters
    ----------
    X : np.array
        x axis values
    Y : np.array
        y axis values
    sigma : np.array
        standard deviation of y axis values
    x_label : str
        x axis label
    y_label : str
        y axis label

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        _description_
    """
    data = pd.DataFrame(data = {
        x_label: X, y_label: Y, 'sigma': sigma
    })
    data['sigma_min'] = data[y_label] - data.sigma
    data['sigma_max'] = data[y_label] + data.sigma

    plot = (
        p9.ggplot(data, p9.aes(x=x_label)) +
        p9.geom_line(p9.aes(y=y_label), color = '#1E90FF') +
        p9.geom_ribbon(p9.aes(y = y_label, ymin = 'sigma_min', ymax = 'sigma_max'), alpha = 0.2) +
        p9.xlim([0, data.shape[0]]) +
        # p9.ylim(10**min_log, round(data[y_label].max(), 1)) +
        # p9.scale_y_log10() +
        p9.theme_bw() +
        p9.theme(figure_size=(width, height))
    )
    return plot

def test_plot(Y_test: np.array, x_label: str, y_label:str,
                width: int = 5, height: int = 3):
    """ Testing on the dataset is only done every 50 epochs or so with nan in the data. Returns a line p9.ggplot of the
    test results where nan are filtered out at the beginning."""

    x = np.arange(Y_test.shape[0])

    x_filt = x[~np.isnan(Y_test)]
    y_filt = Y_test[~np.isnan(Y_test)]

    data = pd.DataFrame(data = {x_label: x_filt, y_label: y_filt})

    plot = (
        p9.ggplot(data, p9.aes(x = x_label)) +
        p9.geom_line(p9.aes(y= y_label), color = '#1E90FF') +
        p9.xlim([0, data.shape[0]]) +
        p9.theme_bw() +
        p9.theme(figure_size=(width, height))
        )

    return plot

def ind_TF_corr_plot(corr_TF_data: np.array, TF_name_list, x_label: str = 'Epochs', y_label : str = 'Correlation'):
    """ corr_TF_data is an array of the shape [num of epochs, num of TF] in which only specific epochs (every 50 or so) are tested.
    Returns a p9.ggplot object that filters out the nan and plots each TF correlation line plot as a separate color.
    """

    x = np.arange(corr_TF_data.shape[0])

    # mask for filtering if there is an nan in the epoch (indicating that it was not a test epoch)
    na_mask = np.any(np.isnan(corr_TF_data), axis = 1)

    x_filt = x[~na_mask]
    y_filt = corr_TF_data[~na_mask]

    x_filt_df = pd.DataFrame(data = {x_label: x_filt})
    y_filt_df = pd.DataFrame(y_filt, columns = TF_name_list)

    data = pd.concat([x_filt_df, y_filt_df], axis = 1)

    melted_data = data.melt(id_vars = x_label, var_name='Transcription Factor', value_name = y_label)
    print(melted_data)
    plot = (
        p9.ggplot(melted_data, p9.aes(x = x_label, y = y_label, color = 'Transcription Factor')) +
        p9.geom_line() +
        p9.theme_bw()
        )

    return plot

def line_plot(X: np.array, Y : np.array, x_label: str, y_label: str,
            width: int = 5, height: int = 3):
    """ X and Y are np.arrays containing the dataset for the x and y values
    of corresponding values. Y may consist of NaN values, so these points will
    be removed. Returns a p9.ggplot object of a line plot between x and y. """
    y_filt = Y[~np.isnan(Y)]
    x_filt = X[~np.isnan(Y)]

    data = pd.DataFrame(data = {x_label: x_filt, y_label: y_filt})

    plot = (
        p9.ggplot(data, p9.aes(x = x_label)) +
        p9.geom_line(p9.aes(y= y_label), color = '#1E90FF') +
        p9.xlim([0, data.shape[0]]) +
        p9.theme_bw() +
        p9.theme(figure_size=(width, height))
        )

    return plot

test = np.nan*np.ones((100, 100))

for epoch in range(100):
    if epoch % 10 == 0:
        lt = np.random.rand(100)
        test[epoch] =  lt

print(test)

plt = ind_TF_corr_plot(test, TF_name_list=[f'{i}' for i in range(100)])

plt.draw(show = True)
