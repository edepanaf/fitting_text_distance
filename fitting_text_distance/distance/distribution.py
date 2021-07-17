# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.tools.matrix_operations import *


class Distribution:
    """Probability distribution on a finite sequence of floats.

    Attributes
    ----------
    values: list or tuple of floats
    probabilities: list or tuple of floats summing to '1.'

    Methods
    -------
    get_moment(order: int) -> float
        Returns the moment of order 'order' of self.
    get_mean() -> float
    get_variance() -> float
    """

    def __init__(self, values, probabilities):
        """Creates a 'Distribution' object from two lists: values and associated probabilities.

        Parameters
        ----------
        values: list or tuple of floats
        probabilities: list or tuple of floats summing to '1.'
        """
        if len(values) != len(probabilities):
            raise ValueError
        self.values = values
        self.probabilities = probabilities
        self.moments = dict()

    def __len__(self):
        return len(self.values)

    def get_moment(self, order):
        """Returns the moment of order 'order' of self.

        Parameters
        ----------
        order: int

        Returns
        -------
        Sum of self.probabilities[i] * self.values[i] ** order for i from 0 to len(self).
        """
        return scalar_product(self.probabilities, coefficient_wise_power_from_vector(self.values, order))

    def get_mean(self):
        return self.get_moment(1)

    def get_variance(self):
        return self.get_moment(2) - self.get_moment(1)**2
