# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from tests.random_vectors_and_probabilities import *
from fitting_text_distance.fitting_vectorization.fitting_vectorization import *


bag_collection = ['aa', 'ab', 'bbb']
vectorize = FittingVectorization(bag_collection)
saved_item_weights_vector = vectorize.item_weights_vector.copy()
saved_bag_weights_vector = vectorize.bag_weights_vector.copy()


def reinitialize_vectorize():
    vectorize.item_weights_vector = saved_item_weights_vector
    vectorize.bag_weights_vector = saved_bag_weights_vector


def function(v0, v1):
    return scalar_product(coefficient_wise_vector_product(v0, v0), v1) + scalar_product(v1, v1)


def partial_gradient0(v0, v1):
    return 2 * coefficient_wise_vector_product(v0, v1)


def partial_gradient1(v0, v1):
    return coefficient_wise_vector_product(v0, v0) + 2 * v1


function.partial_gradients = (partial_gradient0, partial_gradient1)


class TestFittingVectorization(unittest.TestCase):

    def test_tfidf_item_weights(self):
        vectorization0 = FittingVectorization(['aa', 'ab', 'bbb', 'aaa'], tfidf=True)
        expected_item_weights_vector0 = make_vector([np.log(4 / 3), np.log(4 / 2)])
        expected_item_weights_vector1 = make_vector([np.log(4 / 2), np.log(4 / 3)])
        self.assertTrue(are_almost_colinear_vectors(vectorization0.item_weights_vector,
                                                    expected_item_weights_vector0) or
                        are_almost_colinear_vectors(vectorization0.item_weights_vector,
                                                    expected_item_weights_vector1))

    def test_linear_fit_from_form_arguments_interval(self):
        bag_arguments = ({'aa', 'ab'}, {'ab', 'bbb'})
        current_value = function(*(vectorize(argument) for argument in bag_arguments))
        target_value = current_value - 0.2
        interval = (0., target_value)
        for _ in range(50):
            vectorize.linear_fit_from_form_arguments_interval(function, bag_arguments, interval,
                                                              speed=1., ratio_item_bag_fitting=0.5)
        new_value = function(*(vectorize(argument) for argument in bag_arguments))
        self.assertTrue(np.isclose(new_value, target_value))
        reinitialize_vectorize()

    def test_linear_fit_from_form_arguments_target(self):
        bag_arguments = ({'aa', 'ab'}, {'ab', 'bbb'})
        current_value = function(*(vectorize(argument) for argument in bag_arguments))
        target_value = current_value + 0.4
        for _ in range(50):
            vectorize.linear_fit_from_form_arguments_target(function, bag_arguments, target_value,
                                                            speed=1., ratio_item_bag_fitting=0.5)
        new_value = function(*(vectorize(argument) for argument in bag_arguments))
        self.assertTrue(np.isclose(new_value, target_value))
        reinitialize_vectorize()

    def test_partial_gradients_from_function_on_vectorizations(self):
        number_of_try = 100
        bags0 = {'aa', 'ab'}
        bags1 = {'ab', 'bbb'}
        vectorization00 = vectorize(bags0)
        vectorization01 = vectorize(bags1)
        for _ in range(number_of_try):
            item_perturbation = random_vector(len(vectorize.item_weights_vector), lower_bound=0., upper_bound=0.001)
            bag_perturbation = random_vector(len(vectorize.bag_weights_vector), lower_bound=0., upper_bound=0.001)
            vectorize.update_item_weights_from_perturbation(item_perturbation)
            vectorize.update_bag_weights_from_perturbation(bag_perturbation)
            vectorization10 = vectorize(bags0)
            vectorization11 = vectorize(bags1)
            reinitialize_vectorize()
            item_partial_gradient, bag_partial_gradient = \
                vectorize.partial_gradients_from_form_on_vectorizations(function, (bags0, bags1))
            computed = (function(vectorization00, vectorization01)
                        - function(vectorization10, vectorization11)
                        - scalar_product(item_partial_gradient, item_perturbation)
                        - scalar_product(bag_partial_gradient, bag_perturbation))
            # print('yop' + str(computed))
            self.assertTrue(computed <= 0.0001)

    def test_partial_jacobian_duals_from_bags(self):
        number_of_try = 100
        bags = {'aa', 'ab'}
        vectorization0 = vectorize(bags)
        item_jacobian_dual, bag_jacobian_dual = vectorize.partial_jacobian_duals_from_bags(bags)
        for _ in range(number_of_try):
            item_perturbation = random_vector(len(vectorize.item_weights_vector), lower_bound=0., upper_bound=0.001)
            bag_perturbation = random_vector(len(vectorize.bag_weights_vector), lower_bound=0., upper_bound=0.001)
            vector = random_vector(len(vectorization0))
            vectorize.update_item_weights_from_perturbation(item_perturbation)
            vectorize.update_bag_weights_from_perturbation(bag_perturbation)
            vectorization1 = vectorize(bags)
            reinitialize_vectorize()
            computed = (scalar_product(vector, vectorization1 - vectorization0)
                        - scalar_product(item_jacobian_dual(vector), item_perturbation)
                        - scalar_product(bag_jacobian_dual(vector), bag_perturbation))
            self.assertTrue(computed <= 0.0001)

    def test_vectors_to_reach_target_from_partial_gradients__reaches_target(self):
        number_of_iterations = 10
        for _ in range(number_of_iterations):
            size = 2
            length = 2
            partial_gradients = tuple(random_vector(length) for _ in range(size))
            balances = random_probabilities_tuple_from_length(size)
            distance_to_target = random_float(-4., 4.)
            vectors = vectors_to_reach_target_from_partial_gradients(partial_gradients, balances, distance_to_target)
            self.assertAlmostEqual(sum_of_scalar_products(partial_gradients, vectors), distance_to_target)
            for _ in range(number_of_iterations):
                vectors_candidate = tuple(random_vector(length) for _ in range(size))
                renormalization_factor = distance_to_target / sum_of_scalar_products(partial_gradients,
                                                                                     vectors_candidate)
                for candidate in vectors_candidate:
                    candidate *= renormalization_factor
                candidate_norm = sum_weighted_squared_norms(balances, vectors_candidate)
                computed_norm = sum_weighted_squared_norms(balances, vectors)
                self.assertTrue(candidate_norm > computed_norm)

    def test_side_effect_rescale_vector_from_perturbation(self):
        perturbation = make_vector([0.5])
        vector = make_vector([0.7])
        copy_perturbation = perturbation.copy()
        copy_vector = vector.copy()
        _ = rescale_vector_from_perturbation(perturbation, speed=0.7)
        self.assertTrue(are_almost_equal_vectors(vector, copy_vector))
        self.assertTrue(are_almost_equal_vectors(perturbation, copy_perturbation))

    def test_rescale_vector_from_perturbation(self):
        number_of_iteration = 100
        length = 5
        large_speed = 1.
        small_speed = 0.7
        one_vector = one_vector_from_length(length)
        for _ in range(number_of_iteration):
            perturbation = random_vector(length, lower_bound=-1.1, upper_bound=2.)
            rescale_vector = rescale_vector_from_perturbation(perturbation, speed=large_speed)
            self.assertTrue(are_almost_equal_vectors(rescale_vector_from_perturbation(perturbation, speed=small_speed),
                                                     one_vector + small_speed * (rescale_vector - one_vector)))
            if min(perturbation) > -1.:
                self.assertTrue(are_almost_equal_vectors(rescale_vector, one_vector + perturbation))

    def test_dot_dot_matrix_dot_products(self):
        v0 = random_float()
        v1 = random_float()
        m = random_float()
        v2 = random_float()
        v3 = random_float()
        expected = make_vector([v0 * v1 * m * v2 * v3])
        vector0 = make_vector([v0])
        vector1 = make_vector([v1])
        vector2 = make_vector([v2])
        vector3 = make_vector([v3])
        matrix = m * matrix_from_bags_and_index_maps({'a'}, {'a': 0}, {'a': 0})
        computed = dot_dot_matrix_dot_products(vector0, vector1, matrix, vector2, vector3)
        self.assertTrue(are_almost_equal_vectors(expected, computed))

    def test_closest_point_from_interval(self):
        interval = (-2, 4)
        self.assertEqual(closest_point_from_interval(-3, interval), -2)
        self.assertEqual(closest_point_from_interval(-2, interval), -2)
        self.assertEqual(closest_point_from_interval(5, interval), 4)
        self.assertEqual(closest_point_from_interval(1, interval), 1)

    def test_transpose_from_list_matrix(self):
        list_matrix = [[1, 2, 3], [4, 5, 6]]
        expected = [[1, 4], [2, 5], [3, 6]]
        computed = transpose_from_list_matrix(list_matrix)
        self.assertEqual(expected, computed)

    def test_prod(self):
        my_list = [1, 2, 3, 4]
        expected = 1 * 2 * 3 * 4
        computed = prod(my_list)
        self.assertEqual(expected, computed)


def sum_weighted_squared_norms(balances, vectors):
    return sum((balances[index] * norm_from_vector(vectors[index]))**2 for index in range(len(balances)))


def sum_of_scalar_products(vectors0, vectors1):
    return sum(scalar_product(vectors0[index], vectors1[index]) for index in range(len(vectors0)))


if __name__ == '__main__':
    unittest.main()
