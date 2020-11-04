# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from itertools import chain
from fitting_text_distance.fitting_vectorization.vector_from_iterable import VectorFromIterable


class Vectorization:

    def __init__(self, bags, item_to_weight=None, tfidf=False):
        self.bag_vector_representation = VectorFromIterable(bags, silent=False,
                                                            weight_from_key_and_multiplicity=lambda _, __: 1.)
        self.item_vector_representation = VectorFromIterable(chain(*bags), silent=True,
                                                             weight_from_key_and_multiplicity=lambda _, __: 1.)
        self.item_bag_matrix = matrix_from_bags_and_index_maps(
            bags, self.item_vector_representation.index_map, self.bag_vector_representation.index_map)
        self.bag_weights_vector = None
        self.initialize_bag_weights_vector(bags, tfidf)
        self.item_weights_vector = None
        self.initialize_item_weights_vector(item_to_weight, tfidf)

    def __call__(self, bags):
        vector = self.bag_vector_representation(bags)
        return dot_matrix_dot_products(self.item_weights_vector, self.item_bag_matrix,
                                       self.bag_weights_vector, vector)

    def items(self):
        return self.item_vector_representation.keys()

    def bags(self):
        return self.bag_vector_representation.keys()

    def set_bag_weights(self, bag_to_weight, silent=False):
        self.bag_weights_vector = self.bag_vector_representation(bag_to_weight, silent=silent)

    def set_item_weights(self, item_to_weight, silent=True):
        self.item_weights_vector = self.item_vector_representation(item_to_weight, silent=silent)

    def get_bag_weights(self):
        return self.bag_vector_representation.dict_from_vector(self.bag_weights_vector)

    def get_item_weights(self):
        return self.item_vector_representation.dict_from_vector(self.item_weights_vector)

    def get_bag_weight(self, bag):
        index = self.bag_vector_representation.index_map.get(bag, None)
        if index is None:
            raise ValueError('This bag does not belong to the initial collection.')
        return self.bag_weights_vector[index]

    def get_item_weight(self, item):
        index = self.item_vector_representation.index_map.get(item, None)
        if index is None:
            raise ValueError('This item does not appear in the initial collection.')
        return self.item_weights_vector[index]

    def count_bags_containing_item(self, item):
        if item not in self.items():
            return 0
        return count_nonzero_entries_in_matrix_row(self.item_bag_matrix,
                                                   self.item_vector_representation.index_map[item])

    def initialize_bag_weights_vector(self, bags, tfidf):
        if (not isinstance(bags, dict)) and tfidf:
            bags = self.tfidf_bag_weights()
        self.set_bag_weights(bags)

    def initialize_item_weights_vector(self, item_to_weight, tfidf):
        if item_to_weight is None:
            if tfidf:
                item_to_weight = self.tfidf_item_weights()
            else:
                item_to_weight = self.items()
        self.set_item_weights(item_to_weight)

    def tfidf_bag_weights(self):
        return {bag: 1 / len(bag) for bag in self.bags()}

    def tfidf_item_weights(self):
        number_of_bags = len(self.bags())
        # Use of log_of_ratio_zero_if_null_denominator to handle the case
        # where the only bag containing an item has been removed (operation currently not supported).
        return {item: log_of_ratio_zero_if_null_denominator(number_of_bags, self.count_bags_containing_item(item))
                for item in self.items()}


def log_of_ratio_zero_if_null_denominator(numerator, denominator):
    if denominator == 0:
        return 0.
    return np.log(numerator / denominator)
