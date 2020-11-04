# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest

iterables = ['banana', 'ananas', 'base']
item_to_index = {'b': 0, 'a': 1, 'n': 2, 's': 3, 'e': 4}
iterable_to_index = {'banana': 0, 'ananas': 1, 'base': 2}
matrix = csr_matrix([[1, 0, 1], [3, 3, 1], [2, 2, 0], [0, 1, 1], [0, 0, 1]])


class TestMatrixOperations(unittest.TestCase):

    def test_matrix_from_bags_and_index_maps(self):
        computed = matrix_from_bags_and_index_maps(iterables, item_to_index, iterable_to_index)
        expected = matrix
        row_number, column_number = computed.get_shape()
        self.assertEqual((row_number, column_number), (5, 3), 'wrong shape')
        for i in range(row_number):
            for j in range(column_number):
                self.assertEqual(computed[i, j], expected[i, j])

    def test_count_nonzero_entries_in_matrix_row(self):
        row_number, _ = matrix.get_shape()
        self.assertEqual(row_number, 5)
        computed = [count_nonzero_entries_in_matrix_row(matrix, i) for i in range(row_number)]
        expected = [2, 3, 2, 2, 1]
        for i in range(row_number):
            self.assertEqual(computed[i], expected[i])

    def test_rescale_vector_to_satisfy_lower_negative_bound(self):
        vector = make_vector([6, 2, 4, 8])
        self.assertTrue(are_equal_vectors(rescale_vector_to_satisfy_lower_negative_bound(vector, -1), vector))
        vector = make_vector([6, -2, 4, 8])
        self.assertTrue(are_equal_vectors(rescale_vector_to_satisfy_lower_negative_bound(vector, -1),
                                          make_vector([3, -1, 2, 4])))

    def test_side_effect_rescale_vector_to_satisfy_lower_negative_bound(self):
        vector = make_vector([6, -2, 4, 8])
        copy_vector = vector.copy()
        _ = rescale_vector_to_satisfy_lower_negative_bound(vector, -1)
        self.assertTrue(are_almost_equal_vectors(vector, copy_vector))

    def test_has_nonnegative_coefficients(self):
        vector = make_vector([0.1, 0.3, 0.4, 0.])
        self.assertTrue(contains_only_nonnegative_coefficients(vector))
        vector = make_vector([0.3, -0.1, 0.5, 0.])
        self.assertFalse(contains_only_nonnegative_coefficients(vector))

    def test_concatenate_vector(self):
        v0 = make_vector([1, 2, 3])
        v1 = make_vector([4, 5])
        expected = make_vector([1, 2, 3, 4, 5])
        computed = concatenate_vectors(v0, v1)
        self.assertTrue(are_equal_vectors(expected, computed))

    def test_dot_matrix_dot_products(self):
        v0 = make_vector([1, 2, 3])
        v1 = make_vector([4, 5])
        v2 = make_vector([2, 1])
        m = matrix_from_bags_and_index_maps({'aab', 'bcc'}, {'a': 0, 'b': 1, 'c': 2}, {'aab': 0, 'bcc': 1})
        # matrix [[2, 0], [1, 1], [0, 2]]
        vv0 = v0.copy()
        vv1 = v1.copy()
        vv2 = v2.copy()
        mm = m.copy()
        expected = make_vector([16, 26, 30])
        computed = dot_matrix_dot_products(v0, m, v1, v2)
        self.assertTrue(are_equal_vectors(computed, expected))
        self.assertTrue(are_equal_vectors(v0, vv0))
        self.assertTrue(are_equal_vectors(v1, vv1))
        self.assertTrue(are_equal_vectors(v2, vv2))


if __name__ == '__main__':
    unittest.main()
