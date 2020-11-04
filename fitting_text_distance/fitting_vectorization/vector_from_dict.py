# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.tools.matrix_operations import *


class VectorFromDict:
    """ Define a callable object that turns dictionaries into vectors.

        Parameters
        ----------
        initial_iterable : iterable
            The set of all items that can appear in the dictionaries that will be turned into vectors.
        silent : bool (default False)
            Whether 'self.__call__' raises a 'ValueError' exception by default
            when a key of the input dictionary does not belong to the initial iterable.

        Attributes
        ----------
        index_map : dict[item -> int]
            Associate to each item its index. Indices are consecutive integers starting at '0'.
        silent : bool
            The default value of the keyword parameter 'silent' of the method '__call__'.

        Examples
        --------
        Create a 'VectorRepresentation' object.

            >>> vector_from_dict = VectorFromDict(['a', 'b', 'c'])

        Take the keys and values of the 'index_map' attribute.

            >>> keys, values = vector_from_dict.index_map.keys(), vector_from_dict.index_map.values()

        The keys are the items of the iterable and the values are consecutive integers starting at '0'.

            >>> assert(set(keys) == {'a', 'b', 'c'})
            >>> assert(set(values) == {0, 1, 2})
        """

    def __init__(self, initial_iterable, silent=False):
        self.index_map = index_map_from_iterable(initial_iterable)
        self.silent = silent

    def __call__(self, key_to_weight, silent=None):
        """ Turn a dictionary associating keys to their weights into a vector.

        Parameters
        ----------
        key_to_weight : dict
        silent : bool
            Whether the ValueError exception is raised if some keys did not belong to the initial iterable.

        Returns
        -------
        vector : numpy.array

        Raises
        ------
        ValueError
            If silent is False and some keys did not belong to the initial iterable.

        Examples
        --------
        Create a 'VectorRepresentation' object.

            >>> vector_from_dict = VectorFromDict(['a', 'b', 'c'])

            Transform a dict into a vector.

            >>> assert(set(vector_from_dict({'b': 3.14, 'c': 1.})) == {0., 3.14, 1.})
        """
        if silent is None:
            silent = self.silent
        return vector_from_index_and_value_maps(self.index_map, key_to_weight, silent=silent)

    def dict_from_vector(self, vector):
        return {item: vector[index] for item, index in self.index_map.items()}

    def keys(self):
        return self.index_map.keys()


def index_map_from_iterable(iterable):
    return {item: index for index, item in enumerate(set(iterable))}


def vector_from_index_and_value_maps(to_index: dict, to_value, silent=False):
    vector = zero_vector_from_length(len(to_index))
    for key, value in to_value.items():
        index = to_index.get(key, None)
        if index is None and (not silent):
            raise ValueError('This item has no index.')
        if index is not None:
            vector[index] = value
    return vector
