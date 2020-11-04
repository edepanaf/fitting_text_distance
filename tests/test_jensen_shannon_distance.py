# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from tests.random_vectors_and_probabilities import *
from fitting_text_distance.distance.jensen_shannon_distance import *


js_distance = JensenShannonDistance()


class TestJensenShannonDistance(unittest.TestCase):

    def test_jensen_shannon_distance(self):
        length = 5
        vector0 = random_vector(length)
        vector1 = random_vector(length)
        probabilities0 = probabilities_from_vector(vector0)
        probabilities1 = probabilities_from_vector(vector1)
        mixture = mixture_from_probabilities(probabilities0, probabilities1)
        entropy0 = entropy_from_probabilities(probabilities0)
        entropy1 = entropy_from_probabilities(probabilities1)
        entropy_mixture = entropy_from_probabilities(mixture)
        self.assertAlmostEqual(js_distance(vector0, vector1), np.sqrt(entropy_mixture - 0.5 * (entropy0 + entropy1)))

    def test_intuition_jensen_shannon_distance(self):
        vector0 = make_vector([0.9, 0.8, 1.1])
        vector1 = make_vector([0.8, 1.1, 0.9])
        vector2 = make_vector([0.1, 1., 0.3])
        self.assertTrue(js_distance(vector0, vector1) < js_distance(vector0, vector2))
        self.assertTrue(js_distance(vector0, vector1) < js_distance(vector1, vector2))

    def test_proportional_jensen_shannon_distance(self):
        vector0 = make_vector([1, 1, 1])
        vector1 = make_vector([0, 1, 2])
        vector2 = 2 * vector0
        vector3 = 3 * vector1
        self.assertAlmostEqual(js_distance(vector0, vector1), js_distance(vector2, vector3))

    def test_side_effect_jensen_shannon_distance(self):
        vector = make_vector([1, 1, 1])
        copy_vector = vector.copy()
        _ = js_distance(vector, vector)
        self.assertTrue(are_equal_vectors(vector, copy_vector))

    def test_disjoint_jensen_shannon_distance(self):
        vector0 = make_vector([0, 1, 0, 1])
        vector1 = make_vector([1, 0, 1, 0])
        self.assertAlmostEqual(js_distance(vector0, vector1), 1.)

    def test_identical_jensen_shannon_distance(self):
        vector = make_vector([1, 1, 1])
        self.assertAlmostEqual(js_distance(vector, vector), 0.)

    def test_entropy_from_probabilities(self):
        vector = make_vector([3, 4, 3])
        expected = make_vector([0.3, 0.4, 0.3])
        computed = probabilities_from_vector(vector)
        self.assertTrue(are_almost_equal_vectors(expected, computed))

    def test_mixture_from_probabilities(self):
        length = 5
        probabilities0 = make_vector(random_probabilities_tuple_from_length(length))
        probabilities1 = make_vector(random_probabilities_tuple_from_length(length))
        mixture = mixture_from_probabilities(probabilities0, probabilities1)
        self.assertEqual(len(mixture), length)

    def test_positivity_jensen_shannon_distribution_from_probabilities(self):
        vector0 = make_vector([1, 1, 1])
        vector1 = make_vector([0, 1, 2])
        probabilities0 = probabilities_from_vector(vector0)
        probabilities1 = probabilities_from_vector(vector1)
        js_distribution = jensen_shannon_distribution_from_probabilities(probabilities0, probabilities1)
        self.assertTrue(js_distribution.get_mean() > 0)

    def test_length_jensen_shannon_distribution_from_probabilities(self):
        length = 5
        probabilities0 = make_vector(random_probabilities_tuple_from_length(length))
        probabilities1 = make_vector(random_probabilities_tuple_from_length(length))
        js_distribution = jensen_shannon_distribution_from_probabilities(probabilities0, probabilities1)
        self.assertEqual(len(js_distribution.probabilities), 2 * length)
        self.assertEqual(len(js_distribution.values), 2 * length)


if __name__ == '__main__':
    unittest.main()
