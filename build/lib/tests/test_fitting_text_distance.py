# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from fitting_text_distance.oracle_claim import OracleClaim


class TestFittingDistanceOnTextCollections(unittest.TestCase):

    def test_tfidf(self):
        distance = FittingTextDistance(['aa', 'ab', 'bbb', 'aaa'], max_factor_length=1)
        expected_item_weights_vector = make_vector([np.log(4 / 3), np.log(4 / 2)])
        computed = make_vector([distance.get_factor_weight('a'), distance.get_factor_weight('b')])
        self.assertFalse(is_almost_zero_vector(computed))
        self.assertTrue(are_almost_colinear_vectors(computed, expected_item_weights_vector))

    def test_fit_of_fitting_text_distance(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        distance = FittingTextDistance(text_collection)
        oracle_claim = OracleClaim(({text1}, {text2}), (0.1, 0.2))
        old_distance12 = distance({text1}, {text2})
        distance.fit({oracle_claim})
        new_distance12 = distance({text1}, {text2})
        self.assertTrue(old_distance12 > new_distance12)

    def test_type_text_distance(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        distance = FittingTextDistance(text_collection)
        self.assertTrue(isinstance(distance({text0, text1}, {text1, text2}), float))

    def test_type_bag_of_factors_from_text(self):
        bag = bag_of_factors_from_text('wefwef')
        self.assertTrue(isinstance(bag, tuple))

    def test_bag_of_factors_from_text(self):
        text = 'ab c  ab '
        bag = bag_of_factors_from_text(text, max_factor_length=3)
        expected = {'a': 2, 'ab': 2, 'ab ': 2, 'b': 2, 'b ': 2, 'b c': 1, ' ': 4, ' c': 1,
                    ' c ': 1, 'c': 1, 'c ': 1, 'c  ': 1, '  ': 1, '  a': 1, ' a': 1, ' ab': 1}
        computed = dict()
        for factor in bag:
            computed[factor] = computed.get(factor, 0) + 1
        self.assertEqual(computed, expected)

    def test_clean_text(self):
        text = 'A;b_C'
        clean = 'a b c'
        self.assertEqual(clean_text(text), clean)

    def test_clean_letter(self):
        self.assertEqual(clean_letter('a'), 'a')
        self.assertEqual(clean_letter(','), ' ')

    def test_non_existent_weight_from_factor(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        distance = FittingTextDistance(text_collection)
        self.assertEqual(distance.get_factor_weight('wef'), 0.)

    def test_input_bag_dictionary(self):
        bag_to_weight = {'aa': 1., 'ab': 2., 'abc': 3.}
        distance = FittingTextDistance(bag_to_weight)
        self.assertEqual(distance.get_text_weight('aa'), 1.)
        self.assertEqual(distance.get_text_weight('ab'), 2.)
        self.assertEqual(distance.get_text_weight('abc'), 3.)

    def test_input_factor_dictionary(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        factor_to_weight = {'a': 1., 'b': 2., 'ba': 3., 'c': 28.}
        distance = FittingTextDistance(text_collection, factor_to_weight=factor_to_weight)
        self.assertEqual(distance.get_factor_weight('a'), 1.)
        self.assertEqual(distance.get_factor_weight('ba'), 3.)
        self.assertEqual(distance.get_factor_weight('d'), 0.)
        self.assertEqual(distance.get_factor_weight('c'), 0.)


# @pytest.fixture
# def response():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
#
#
# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string
