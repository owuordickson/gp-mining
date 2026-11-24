"""Regression checks for comparing gradual-pattern descriptor profiles."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from so4gp import gp_spider_plot


def test_descriptor_spokes_have_independent_scales():
    parameters = ["density", "connectivity", "rank_dispersion"]
    first = pd.DataFrame([[0.1, 1, 10], [0.3, 3, 30]], columns=parameters)
    second = pd.DataFrame([[0.5, 5, 50], [0.7, 7, 70]], columns=parameters)

    figure = gp_spider_plot([first, second], ["Rainfall", "EVI"], parameters)
    try:
        profiles = {
            line.get_label(): np.linalg.norm(line.get_xydata(), axis=1)
            for line in figure.axes[0].lines
            if line.get_label() in {"Rainfall", "EVI"}
        }
        # The descriptors differ only in their units, so all spokes should
        # place the same group at the same distance from the center.
        np.testing.assert_allclose(profiles["Rainfall"], profiles["Rainfall"][0])
        np.testing.assert_allclose(profiles["EVI"], profiles["EVI"][0])
        assert profiles["Rainfall"][0] < profiles["EVI"][0]
    finally:
        plt.close(figure)


def test_single_pattern_profiles_keep_observed_descriptors_visible():
    parameters = ["support", "connectivity", "avg_separation"]
    first = pd.DataFrame([[0.5, 1, np.nan]], columns=parameters)
    second = pd.DataFrame([[0.5, 3, 2.0]], columns=parameters)

    figure = gp_spider_plot([first, second], ["Rainfall", "EVI"], parameters)
    try:
        lines = {line.get_label(): line.get_xydata() for line in figure.axes[0].lines}
        assert np.isfinite(lines["Rainfall"][:2]).all()
        assert np.isnan(lines["Rainfall"][2]).all()
        assert np.isfinite(lines["EVI"]).all()
        figure.canvas.draw()
    finally:
        plt.close(figure)
