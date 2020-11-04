# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse import lil_matrix


def matrix_from_bags_and_index_maps(bags, item_to_index: dict, bag_to_index: dict) -> csr_matrix:
    matrix = lil_matrix((len(item_to_index), len(bag_to_index)), dtype='int')
    for bag in bags:
        for item in bag:
            matrix[item_to_index[item], bag_to_index[bag]] += 1
    return matrix.tocsr()


def count_nonzero_entries_in_matrix_row(matrix, row_index):
    row = matrix.getrow(row_index)
    return row.getnnz()


def scalar_product(vector0, vector1):
    return np.dot(vector0, vector1)


def normalize(vector):
    norm = norm_from_vector(vector)
    if norm == 0.:
        return vector
    return vector / norm


def is_zero_vector(vector):
    return not np.any(vector)


def norm_from_vector(vector):
    return np.sqrt(scalar_product(vector, vector))


def coefficient_wise_vector_product(vector0: np.ndarray, vector1: np.ndarray) -> np.ndarray:
    return np.multiply(vector0, vector1)


def matrix_vector_product(matrix: csr_matrix, vector: np.ndarray) -> np.ndarray:
    return matrix.dot(vector)


def dot_matrix_dot_products(dot_vector0, matrix, dot_vector1, vector):
    vector = coefficient_wise_vector_product(dot_vector1, vector)
    vector = matrix_vector_product(matrix, vector)
    vector = coefficient_wise_vector_product(dot_vector0, vector)
    return vector


def zero_vector_from_length(length: int) -> np.ndarray:
    return np.zeros(length)


def one_vector_from_length(length: int) -> np.ndarray:
    return np.ones(length)


def rescale_vector_to_satisfy_lower_negative_bound(vector, lower_bound):
    min_element = min(vector)
    if min_element < lower_bound:
        return lower_bound / min_element * vector
    return vector


def transpose_from_matrix(matrix):
    return matrix.transpose()


def make_vector(coefficients):
    return np.array(coefficients)


def are_equal_vectors(vector0, vector1):
    return np.array_equal(vector0, vector1)


def are_almost_equal_vectors(vector0, vector1):
    if len(vector0) != len(vector1):
        return False
    return np.isclose(vector0, vector1).all()


def are_almost_colinear_vectors(vector0, vector1):
    if is_zero_vector(vector0):
        return True
    for index in range(len(vector0)):
        if vector0[index] != 0.:
            return are_almost_equal_vectors(vector0 / vector0[index] * vector1[index], vector1)


def is_almost_zero_vector(vector):
    zero_vector = zero_vector_from_length(len(vector))
    return are_almost_equal_vectors(vector, zero_vector)


def has_nonnegative_coefficients(vector):
    return (vector >= 0.).all()


def contains_only_nonnegative_coefficients(vector):
    return (vector >= 0.).all()


def probabilities_from_vector(vector):
    sum_vector = sum(vector)
    if sum_vector == 0.:
        raise ValueError
    return vector / sum_vector


def information_log_from_vector(vector):
    return np.log2(vector, out=np.zeros_like(vector), where=(vector != 0))


def power_from_vector(vector, power):
    return vector ** power


def concatenate_vectors(vector0, vector1):
    return np.hstack((vector0, vector1))
