# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
import scipy.special
from fitting_text_distance.distance.distribution import *


class TestDistribution(unittest.TestCase):

    def test_distribution(self):
        size = 17
        parameter = 1. / np.pi
        binomial_distribution = binomial_distribution_from_size_and_parameter(size, parameter)
        expected_mean = size * parameter
        expected_variance = size * parameter * (1. - parameter)
        self.assertAlmostEqual(binomial_distribution.get_mean(), expected_mean)
        self.assertAlmostEqual(binomial_distribution.get_variance(), expected_variance)


def binomial_distribution_from_size_and_parameter(size, parameter):
    probabilities = make_vector([scipy.special.binom(size, k) * parameter**k * (1. - parameter)**(size - k)
                                 for k in range(size + 1)])
    values = make_vector(range(size + 1))
    return Distribution(values, probabilities)


if __name__ == '__main__':
    unittest.main()
