# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from fitting_text_distance.fitting_vectorization.vectorization import *


bag_to_weight = {'aa': 1, 'ab': 2, 'bbb': 3}
item_to_weight = {'a': 1, 'b': 2}
vectorization = Vectorization(bag_to_weight, item_to_weight)
bags0 = {'ab'}
bags1 = {'bbb'}


class TestVectorization(unittest.TestCase):

    def test_call(self):
        vector0 = vectorization(bags0)
        vector1 = vectorization(bags1)
        self.assertTrue(are_almost_colinear_vectors(vector0, make_vector([2, 4])) or
                        are_almost_colinear_vectors(vector0, make_vector([4, 2])))
        self.assertTrue(are_almost_colinear_vectors(vector1, make_vector([0, 18])) or
                        are_almost_colinear_vectors(vector1, make_vector([18, 0])))
        self.assertTrue(are_almost_colinear_vectors(vector0 + vector1, vectorization({'ab', 'bbb'})))

    def test_call_dict(self):
        vector = vectorization({'ab': 3., 'bbb': 4.})
        self.assertTrue(are_almost_colinear_vectors(vector, make_vector([6, 84])) or
                        are_almost_colinear_vectors(vector, make_vector([84, 6])))

    def test_no_item_weight(self):
        vectorization0 = Vectorization(bag_to_weight, item_to_weight.keys())
        expected = make_vector([1. for _ in range(len(item_to_weight))])
        self.assertTrue(are_equal_vectors(vectorization0.item_weights_vector, expected))

    def test_no_bag_weight(self):
        vectorization0 = Vectorization(bag_to_weight.keys(), item_to_weight)
        expected = make_vector([1. for _ in range(len(bag_to_weight))])
        self.assertTrue(are_equal_vectors(vectorization0.bag_weights_vector, expected))

    def test_count_bags_containing_item(self):
        computed = vectorization.count_bags_containing_item('a')
        expected = 2
        self.assertEqual(computed, expected)

    def test_input_dictionary_bag(self):
        self.assertEqual(sum(vectorization.bag_weights_vector), 6.)

    def test_bag_weights(self):
        vectorization0 = Vectorization(bag_to_weight, item_to_weight)
        bag_to_weight0 = {'aa': 5, 'ab': 1, 'bbb': 3}
        vectorization0.set_bag_weights(bag_to_weight0)
        computed = vectorization0.get_bag_weights()
        self.assertEqual(bag_to_weight0, computed)

    def test_item_weights(self):
        vectorization0 = Vectorization(bag_to_weight, item_to_weight)
        item_to_weight0 = {'a': 4, 'b': 1}
        vectorization0.set_item_weights(item_to_weight0)
        computed = vectorization0.get_item_weights()
        self.assertEqual(item_to_weight0, computed)

    def test_bag_weight(self):
        self.assertEqual(vectorization.get_bag_weight('ab'), 2)

    def test_exception_bag_weight(self):
        self.assertRaises(ValueError, vectorization.get_bag_weight, 'weee')

    def test_item_weight(self):
        self.assertEqual(vectorization.get_item_weight('b'), 2)

    def test_exception_item_weight(self):
        self.assertRaises(ValueError, vectorization.get_item_weight, 'weee')

    def test_tfidf_item_weights0(self):
        vectorization0 = Vectorization(['aa', 'ab', 'bbb', 'aaa'], tfidf=True)
        expected_item_weights_vector0 = make_vector([np.log(4 / 3), np.log(4 / 2)])
        expected_item_weights_vector1 = make_vector([np.log(4 / 2), np.log(4 / 3)])
        self.assertFalse(is_almost_zero_vector(vectorization0.item_weights_vector))
        self.assertTrue(are_almost_colinear_vectors(vectorization0.item_weights_vector,
                                                    expected_item_weights_vector0) or
                        are_almost_colinear_vectors(vectorization0.item_weights_vector,
                                                    expected_item_weights_vector1))

    def test_tfidf_item_weights1(self):
        bag0 = 'ab'
        bag1 = 'a'
        bag2 = 'aa'
        bag3 = 'aaa'
        bag4 = 'bb'
        collection = {bag0, bag1, bag2, bag3, bag4}
        vectorize = Vectorization(collection, tfidf=True)
        self.assertTrue(vectorize.get_item_weight('a') < vectorize.get_item_weight('b'))

    def test_log_of_ratio_zero_if_null_denominator(self):
        numerator = 5.
        denominator = 2.
        expected = np.log(numerator / denominator)
        computed = log_of_ratio_zero_if_null_denominator(numerator, denominator)
        self.assertAlmostEqual(expected, computed)
        self.assertAlmostEqual(log_of_ratio_zero_if_null_denominator(numerator, 0.), 0.)


if __name__ == '__main__':
    unittest.main()
