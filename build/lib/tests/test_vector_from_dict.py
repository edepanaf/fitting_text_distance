# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest

initial_iterable = 'abcde'
vector_from_dict = VectorFromDict(initial_iterable)
silent_vector_from_dict = VectorFromDict(initial_iterable, silent=True)


class TestVectorRepresentation(unittest.TestCase):

    def test_call(self):
        key_to_weight = {'a': 1., 'c': 2., 'd': 3.}
        vector = vector_from_dict(key_to_weight)
        self.assertEqual(sum(vector), sum(key_to_weight.values()))

    def test_silent(self):
        self.assertRaises(ValueError, vector_from_dict, {'a': 1., 'f': 2.})
        self.assertEqual(sum(silent_vector_from_dict({'a': 1., 'f': 2.})), 1.)
        self.assertRaises(ValueError, silent_vector_from_dict, {'a': 1., 'f': 2.}, silent=False)
        self.assertEqual(sum(vector_from_dict({'a': 1., 'f': 2.}, silent=True)), 1.)

    def test_dict_from_vector(self):
        key_to_weight = {'a': 2., 'd': 3.}
        vector = vector_from_dict(key_to_weight)
        expected = {key: 0. for key in initial_iterable}
        for key, weight in key_to_weight.items():
            expected[key] = weight
        self.assertEqual(vector_from_dict.dict_from_vector(vector), expected)

    def test_keys(self):
        self.assertEqual(set(initial_iterable), set(vector_from_dict.keys()))

    def test_vector_from_index_and_value_maps(self):
        to_index = {'a': 0, 'b': 1, 'c': 2}
        to_value = {'a': 0.1, 'c': 3}
        computed = vector_from_index_and_value_maps(to_index, to_value)
        expected = [0.1, 0., 3.]
        for i in range(len(to_index)):
            self.assertEqual(computed[i], expected[i])

    def test_index_map_from_iterable(self):
        iterable = 'abacbde'
        index_map = index_map_from_iterable(iterable)
        self.assertEqual(set(index_map.keys()), {'a', 'b', 'c', 'd', 'e'})
        self.assertEqual(set(index_map.values()), set(i for i in range(5)))


if __name__ == '__main__':
    unittest.main()
