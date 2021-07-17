# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.tools.matrix_operations import *
from fitting_text_distance.distance.distribution import Distribution
from fitting_text_distance.distance.abstract_distance import AbstractDistance


class JensenShannonDistance(AbstractDistance):
    """Distance on vectors with nonnegative coefficients.

    Any non-zero vector of length 'n' with nonnegative coefficients
    can be turned into a probability distribution on '{0, ... , n-1}'
    by dividing each of its coefficients by their sum.
    The JensenShannonDistance first turns its two vector arguments
    into distributions, then outputs the square root
    of their Jensen-Shannon divergence (defined below),
    which is a metric distance on distributions.

    Let B denote a random variable taking
    value 0 with probability 1/2 and 1 with probability 1/2.
    Let Z denote the a random integer sampled following 'distribution[B]'.
    Then the Jensen-Shannon divergence of 'distribution[0]' and 'distribution[1]'
    is defined as the mutual information between B and Z.

    Methods
    -------
    __call__(vector0: array of floats, vector1: array of floats) -> float
        The Jensen-Shannon distance between the renormalized arguments,
        assuming their coefficient a non-negative and the vectors are non-zero.
    first_partial_gradient(vector0: array of floats, vector1: array of floats) -> float
    second_partial_gradient(vector0: array of floats, vector1: array of floats) -> float
    """

    def __init__(self):
        super().__init__()

    def __call__(self, vector0, vector1):
        probabilities0 = probabilities_from_vector(vector0)
        probabilities1 = probabilities_from_vector(vector1)
        distribution = jensen_shannon_distribution_from_probabilities(probabilities0, probabilities1)
        return np.sqrt(distribution.get_mean())

    def first_partial_gradient(self, vector0, vector1):
        probabilities0 = probabilities_from_vector(vector0)
        probabilities1 = probabilities_from_vector(vector1)
        mixture = mixture_from_probabilities(probabilities0, probabilities1)
        return ((information_log_from_vector(probabilities0) - information_log_from_vector(mixture))
                / 4. / self(vector0, vector1))

    @staticmethod
    def get_variance(vector0, vector1):
        probabilities0 = probabilities_from_vector(vector0)
        probabilities1 = probabilities_from_vector(vector1)
        distribution = jensen_shannon_distribution_from_probabilities(probabilities0, probabilities1)
        return distribution.get_variance()


def jensen_shannon_distribution_from_probabilities(probabilities0, probabilities1):
    mixture = mixture_from_probabilities(probabilities0, probabilities1)
    values0 = information_log_from_vector(probabilities0) - information_log_from_vector(mixture)
    values1 = information_log_from_vector(probabilities1) - information_log_from_vector(mixture)
    values = concatenate_vectors(values0, values1)
    probabilities = concatenate_vectors(0.5 * probabilities0, 0.5 * probabilities1)
    return Distribution(values, probabilities)


def mixture_from_probabilities(probabilities0, probabilities1):
    return 0.5 * probabilities0 + 0.5 * probabilities1


def entropy_from_probabilities(probabilities):
    values = - information_log_from_vector(probabilities)
    return Distribution(values, probabilities).get_mean()
