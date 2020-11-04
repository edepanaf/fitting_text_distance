# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.distance.distribution import Distribution
from fitting_text_distance.distance.abstract_distance import AbstractDistance


class JensenShannonDistance(AbstractDistance):

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
