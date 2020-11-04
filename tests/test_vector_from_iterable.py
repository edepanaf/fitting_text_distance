# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from fitting_text_distance.fitting_vectorization.vector_from_iterable import *


initial_iterable = [0, 1, 2, 3, 4]
vector_from_iterable = VectorFromIterable(initial_iterable)
weight_function = lambda key, multiplicity: key + multiplicity
weighted_vector_from_iterable = VectorFromIterable(initial_iterable, weight_from_key_and_multiplicity=weight_function)


class TestVectorFromIterable(unittest.TestCase):

    def test_dict_call(self):
        key_to_weight = {0: 3., 3: 2.}
        self.assertEqual(sum(vector_from_iterable(key_to_weight)), sum(key_to_weight.values()))
        self.assertEqual(sum(weighted_vector_from_iterable(key_to_weight)), sum(key_to_weight.values()))

    def test_list_call(self):
        keys = [1, 1, 3, 1]
        self.assertEqual(sum(vector_from_iterable(keys)), 4)
        self.assertEqual(sum(weighted_vector_from_iterable(keys)), 8)


if __name__ == '__main__':
    unittest.main()
