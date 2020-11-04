# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from fitting_text_distance.fitting_distance import *
from fitting_text_distance.oracle_claim import OracleClaim


class TestFittingDistance(unittest.TestCase):

    def test_call(self):
        bag0 = 'abb'
        bag1 = 'aa'
        bag2 = 'baa'
        bag3 = 'bbb'
        collection = {bag0, bag1, bag2, bag3}
        distance = FittingDistance(collection)
        self.assertAlmostEqual(distance({bag0, bag1}, {bag1, bag0}), 0.)
        self.assertAlmostEqual(distance({bag1}, {bag3}), 1.)
        self.assertTrue(distance({bag1}, {bag2}) < distance({bag0}, {bag1}))

    def test_fit(self):
        bag0 = 'abb'
        bag1 = 'aa'
        bag2 = 'baa'
        bag3 = 'bbb'
        collection = {bag0, bag1, bag2, bag3}
        distance = FittingDistance(collection)
        old_weight_a = distance.get_item_weight('a')
        distance01 = distance({bag0}, {bag1})
        target_interval = (0., distance01 / 2.)
        distance.fit([OracleClaim(({bag0}, {bag1}), target_interval)])
        new_weight_a = distance.get_item_weight('a')
        self.assertTrue(new_weight_a > old_weight_a)

    def test_exception_weight_from_item(self):
        bag0 = 'ab'
        bag1 = 'aa'
        bag2 = 'baa'
        bag3 = 'bbb'
        collection = {bag0, bag1, bag2, bag3}
        distance = FittingDistance(collection)
        self.assertRaises(ValueError, distance.get_item_weight, 'c')

    def test_tfidf_weights(self):
        bag0 = 'ab'
        bag1 = 'a'
        bag2 = 'aa'
        bag3 = 'aaa'
        bag4 = 'bb'
        collection = {bag0, bag1, bag2, bag3, bag4}
        distance = FittingDistance(collection, tfidf=True)
        self.assertTrue(distance.get_item_weight('a') < distance.get_item_weight('b'))

    def test_weight_from_bag(self):
        bag0 = 'abwwafwaf'
        bag1 = 'awfe'
        collection = {bag0, bag1}
        distance = FittingDistance(collection, tfidf=True)
        self.assertTrue(distance.get_bag_weight(bag0) < distance.get_bag_weight(bag1))

    def test_input_dictionary_bag(self):
        bag_to_weight = {'aa': 1., 'ab': 2., 'abc': 3.}
        distance = FittingDistance(bag_to_weight)
        self.assertEqual(distance.get_bag_weight('aa'), 1.)
        self.assertEqual(distance.get_bag_weight('ab'), 2.)
        self.assertEqual(distance.get_bag_weight('abc'), 3.)


if __name__ == '__main__':
    unittest.main()
