# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import random
from fitting_text_distance.fitting_vectorization.vectorization import Vectorization


DEFAULT_SPEED = 0.3
DEFAULT_NUMBER_OF_GRADIENT_STEPS = 6


class FittingVectorization(Vectorization):

    def __init__(self, bags, item_to_weight=None, tfidf=False):
        super().__init__(bags, item_to_weight=item_to_weight, tfidf=tfidf)

    def fit_from_tuples_of_forms_arguments_intervals(self, forms_arguments_intervals,
                                                     speed=DEFAULT_SPEED, ratio_item_bag_fitting=0.5,
                                                     number_of_gradient_steps=DEFAULT_NUMBER_OF_GRADIENT_STEPS):
        forms_arguments_intervals = forms_arguments_intervals.copy()
        for _ in range(number_of_gradient_steps):
            random.shuffle(forms_arguments_intervals)
            for function, arguments, interval in forms_arguments_intervals:
                self.linear_fit_from_form_arguments_interval(function, arguments, interval, speed=speed,
                                                             ratio_item_bag_fitting=ratio_item_bag_fitting)

    def linear_fit_from_form_arguments_interval(self, function, bag_arguments, target_interval,
                                                speed=1., ratio_item_bag_fitting=0.5):
        current_value = function(*(self(bags) for bags in bag_arguments))
        target = closest_point_from_interval(current_value, target_interval)
        if not np.isclose(current_value, target):
            self.linear_fit_from_form_arguments_target(function, bag_arguments, target, speed=speed,
                                                       ratio_item_bag_fitting=ratio_item_bag_fitting)

    def linear_fit_from_form_arguments_target(self, function, bag_arguments, target, speed=1.,
                                              ratio_item_bag_fitting=0.5):
        """ 'speed' is the fraction of the straight-line distance to the target
        traversed in one gradient descent step. """
        item_and_bag_gradients = self.partial_gradients_from_form_on_vectorizations(function, bag_arguments)
        current_function_value = function(*(self(bags) for bags in bag_arguments))
        item_bag_fitting_balance = (ratio_item_bag_fitting, 1. - ratio_item_bag_fitting)
        item_perturbation, bag_perturbation =\
            vectors_to_reach_target_from_partial_gradients(item_and_bag_gradients,
                                                           item_bag_fitting_balance,
                                                           target - current_function_value)
        self.update_item_weights_from_perturbation(item_perturbation, speed)
        self.update_bag_weights_from_perturbation(bag_perturbation, speed)

    def partial_gradients_from_form_on_vectorizations(self, function, bag_arguments):
        """
        :param function: inputs several vectorizations and outputs a float.
            Has an attribute 'function.partial_gradients' that has the same input as 'function'
            and outputs a list of vectors representing the partial gradients of 'function' at the input.
        :param bag_arguments: a collection of bags '(bag_0, ..., bag_n)' of length the arity of 'function'
        :return: two vectors '(item_partial_gradient, bag_partial_gradient)' satisfying the following condition.
        Let 'f0' denote the 'self' function (fitting_vectorization) for the values
        'i0 := self.item_weights_vector' and 'b0 := self.bag_item_weights_vector'.
        Let 'item_perturbation' and 'bag_perturbation' denote two vectors
        and 'f1' the function 'self' for the parameters
        'i1 := dot_product(vector(1...1) + item_perturbation, i0)',
        'b1 := dot_product(vector(1...1) + bag_perturbation, b0)'.
        Then the vectors '(item_partial_gradient, bag_partial_gradient)' satisfy
        'function(f1(bag_0), ..., f1(bag_n)) = function(f0(bag_0), ..., f0(bag_n))
            + scalar_product(item_partial_gradient, item_perturbation)
            + scalar_product(bag_partial_gradient, bag_perturbation)
            + O(norm(item_perturbation)**2) + norm(bag_perturbation)**2).
        """
        bag_vectors = [self(bags) for bags in bag_arguments]
        function_partial_gradients = [gradient(*bag_vectors) for gradient in function.partial_gradients]
        matrix_of_partial_jacobian_duals = [self.partial_jacobian_duals_from_bags(bags) for bags in bag_arguments]
        matrix_of_partial_jacobian_duals = transpose_from_list_matrix(matrix_of_partial_jacobian_duals)
        return [sum(partial_jacobian_duals[index](function_partial_gradients[index])
                    for index in range(len(function_partial_gradients)))
                for partial_jacobian_duals in matrix_of_partial_jacobian_duals]

    def partial_jacobian_duals_from_bags(self, bags):
        """
        :param bags: bag collection
        :return: pair of functions '(item_jacobian_dual, bag_jacobian_dual)'

        Consider the function 'f0' equal to 'self' (fitting_vectorization of a bag collection) for some values
        'i0 := self.item_weights_vector' and 'b0 := self.bag_weights_vector',
        some perturbations 'item_perturbation' and 'bag_perturbation',
        the function 'f1' equal to 'self' for the values
        'i1 := dot_product(vector(1...1) + item_perturbation, i0)' and
        'b1 := dot_product(vector(1...) + bag_perturbation, b0)'
        and a vector 'r'.
        Then the evaluations of the partial jacobian duals at 'r' are
        'vi := item_jacobian_dual(r)', 'vb := bag_jacobian_dual(r)' and satisfy
        'scalar_product(r, f1(bags)) = scalar_product(r, f0(bags)) + scalar_product(vi, item_perturbation)
                + scalar_product(vb, bag_perturbation) + O(norm(item_perturbation)**2) + O(norm(bag_perturbation)**2)'.
        """
        bag_vector = self.bag_vector_representation(bags)
        vectorization = self(bags)

        def item_jacobian_dual(projection):
            return coefficient_wise_vector_product(vectorization, projection)

        def bag_jacobian_dual(projection):
            return dot_dot_matrix_dot_products(bag_vector,
                                               self.bag_weights_vector,
                                               transpose_from_matrix(self.item_bag_matrix),
                                               self.item_weights_vector,
                                               projection)

        return item_jacobian_dual, bag_jacobian_dual

    def update_item_weights_from_perturbation(self, perturbation, speed=1.):
        rescale_vector = rescale_vector_from_perturbation(perturbation, speed)
        self.item_weights_vector = coefficient_wise_vector_product(rescale_vector, self.item_weights_vector)

    def update_bag_weights_from_perturbation(self, perturbation, speed=1.):
        rescale_vector = rescale_vector_from_perturbation(perturbation, speed)
        self.bag_weights_vector = coefficient_wise_vector_product(rescale_vector, self.bag_weights_vector)


def vectors_to_reach_target_from_partial_gradients(partial_gradients, balances, distance_to_target):
    """
    :param partial_gradients: tuple of vectors '(g_0, ..., g_{n-1})'
    :param balances: tuple of floats '(b_0, ..., b_{n-1})'
    :param distance_to_target: float 'd'
    :return: tuple of vectors '(v_0, ..., v_{n-1})' arg min of '{norm(sum_i b_i * v_i) | sum_i g_i^T * v_i = d}'
    """
    coefficients = [prod(a for a in balances if a != b)**2 for b in balances]
    common_factor = distance_to_target / sum(coefficients[index] * norm_from_vector(partial_gradients[index])**2
                                             for index in range(len(partial_gradients)))
    return [common_factor * coefficients[index] * partial_gradients[index] for index in range(len(partial_gradients))]


def rescale_vector_from_perturbation(perturbation, speed=1.):
    perturbation = rescale_vector_to_satisfy_lower_negative_bound(perturbation, -1.)
    perturbation = speed * perturbation
    one_vector = one_vector_from_length(len(perturbation))
    return one_vector + perturbation


def dot_dot_matrix_dot_products(v0, v1, m, v2, v3):
    v23 = coefficient_wise_vector_product(v2, v3)
    mv23 = matrix_vector_product(m, v23)
    v01 = coefficient_wise_vector_product(v0, v1)
    v01mv23 = coefficient_wise_vector_product(v01, mv23)
    return v01mv23


def closest_point_from_interval(value, interval):
    lower_bound, upper_bound = interval
    if value < lower_bound:
        return lower_bound
    if value > upper_bound:
        return upper_bound
    return value


def transpose_from_list_matrix(matrix):
    return [[matrix[row][column] for row in range(len(matrix))] for column in range(len(matrix[0]))]


def prod(my_list):
    res = 1.
    for value in my_list:
        res *= value
    return res
