# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from itertools import chain
from fitting_text_distance.tools.matrix_operations import *
from fitting_text_distance.fitting_vectorization.vector_from_iterable import VectorFromIterable


class Vectorization:
    """ Callable object that projects collections of immutable iterables into a vector space indexed by their items.

    Let us consider a collection of immutable iterables, that we will call 'bags',
    and a function 'weight' associating to each item
    appearing in the bags a nonnegative float.
    Let 'occurrences(bag, item)' the number of occurrences of 'item' in 'bag'.
    Once an index 'index(item)' has been chosen for each item,
    any bag has a vector representation
    with coefficient 'occurrences(bag, item) * weight(item)'
    at index 'index(item)' for each item.
    The vector representation of a family of bags is then the sum of the vectors of each bag.
    If the bags are weighted, then this linear combination is skewed by those weights.

    By default, all item weights are '1'.
    The user can specify them by provinding a dictionary 'item_to_weight'.
    The item weights can also be computed automatically using tf-idf
    if the optional argument 'tfidf' of '__init__' is set to 'True'.

    The default bag weights are '1'.
    Those weights can be specified by providing
    in '__init__' not a collection of bags,
    but a dictionary associating to each bag its weight.

    Item and bag weights can be modified using the methods
    'set_bag_weights' and 'set_item_weights'.

    Parameters
    ----------
    bags: iterable of immutable iterables or dict(immutable iterable: float)
        The collection of bags of which collections will be turned into vectors.
        If a dictionary is provided, it associates to each bag its weight.
    item_to_weight: (optional) dict
        The weight associated to each item appearing in 'bags'.
    tfidf: (optional) bool, default is 'False'
        Whether the weights should be computed using tf-idf.

    Attributes
    ----------
    bag_vector_representation
    item_vector_representation
    item_bag_matrix
    bag_weights_vector
    item_weights_vector

    Methods
    -------
    items()
    bags()
    set_bag_weights(bag_to_weight, silent=False)
    set_item_weights(item_to_weight, silent=True)
    get_bag_weights()
    get_item_weights()
    get_bag_weight(bag)
    get_item_weight(item)
    count_bags_containing_item(item)

    Examples
    --------
    Definition of our bags and the 'Vectorization' object.
    Note that the bags must be immutable iterables.

    >>> bags = [('x', 'y', 'x'), ('y', 'y', 'y'), ('x', 'x', 'y', 'y')]
    >>> v = Vectorization(bags)

    The bags '('x', 'y', 'x')' and '('y', 'y', 'y')' contain, in total, two 'x' and four 'y',
    so their vectorization

    >>> vector = v([bags[0], bags[1]])

    is an array containing '2' and '4'.

    >>> assert((vector == np.array([2., 4.])).all() or (vector == np.array([4., 2.])).all())

    If we specify the item weights during creation of the 'Vectorization' object
    with 'x' having weight '3' and 'y' weight '7',

    >>> v = Vectorization(bags, item_to_weight={'x': 3, 'y': 7})

    then the vectorization of the bags '('x', 'y', 'x')' and '('y', 'y', 'y')'

    >>> vector = v([bags[0], bags[1]])

    contains '6.' and '28.'.

    >>> assert((vector == np.array([6., 28.])).all() or (vector == np.array([28., 6.])).all())

    The item weights can also be modified after the object creation

    >>> v.set_item_weights({'x': 2., 'y': 3.})
    >>> vector = v([bags[0], bags[1]])
    >>> assert((vector == np.array([4., 12.])).all() or (vector == np.array([12., 4.])).all())

    Weights for the bags can be either specified
    at the creation of the 'Vectorization' object

    >>> v = Vectorization({bags[0]: 10., bags[1]: 1., bags[2]: 5.}, item_to_weight={'x': 3, 'y': 7})
    >>> vector = v([bags[0], bags[1]])
    >>> assert((vector == np.array([91., 60.])).all() or (vector == np.array([60., 91.])).all())

    or modified later

    >>> v.set_bag_weights({bags[0]: 3., bags[1]: 2., bags[2]: 5.})
    """

    def __init__(self, bags, item_to_weight=None, tfidf=False):
        """Create a callable object 'Vectorization'.

        It turns it turns collection of iterables from 'bags' into vectors (numpy array).

        Parameters
        ----------
        bags: iterable of iterables
            The collection of bags (iterables) of which collections will be turned into vectors.
        item_to_weight: (optional) dict

        tfidf: (optional) bool, default is 'False'
        """

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
        """Turn a collection of iterables into a vector.

        Parameters
        ----------
        bags: iterable of iterables

        Returns
        -------
        A vector (numpy array) representing 'bags'

        Examples
        --------
        Definition of our bags and the 'Vectorization' object.
        See the class Docstring for examples.
        """

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
