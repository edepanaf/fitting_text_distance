# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.tools.matrix_operations import *
from fitting_text_distance.distance.abstract_distance import AbstractDistance


class CosineDistance(AbstractDistance):

    def __init__(self):
        super().__init__()

    def __call__(self, vector0, vector1):
        vector0 = normalize(vector0)
        vector1 = normalize(vector1)
        return 1. - scalar_product(vector0, vector1)

    def first_partial_gradient(self, vector0, vector1):
        distance = self(vector0, vector1)
        normalized_vector0 = normalize(vector0)
        normalized_vector1 = normalize(vector1)
        return ((1 - distance) * normalized_vector0 - normalized_vector1) / norm_from_vector(vector0)


""" def test_cosine_distance(self):
        u = make_vector([1., 3., 2.])
        v = make_vector([2., -1., 0.5])
        zero_vector = make_vector([0., 0., 0.])
        self.assertAlmostEqual(cosine_distance(zero_vector, u), 1.)
        expected = 1. - scalar_product(u, v) / norm_from_vector(u) / norm_from_vector(v)
        self.assertAlmostEqual(cosine_distance(u, v), expected)"""
