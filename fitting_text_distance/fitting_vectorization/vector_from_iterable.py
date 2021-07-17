# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from collections import Counter
from fitting_text_distance.fitting_vectorization.vector_from_dict import VectorFromDict


class VectorFromIterable(VectorFromDict):
    """Callable object that turns an iterable into a numpy array containing only '1's and '0's.

    Inherits from 'VectorFromDict'

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

        >>> vector_from_iterable = VectorFromIterable(['a', 'b', 'c'])

    Take the keys and values of the 'index_map' attribute.

        >>> keys, values = vector_from_iterable.index_map.keys(), vector_from_iterable.index_map.values()

    The keys are the items of the iterable and the values are consecutive integers starting at '0'.

        >>> assert(set(keys) == {'a', 'b', 'c'})
        >>> assert(set(values) == {0, 1, 2})
    """

    def __init__(self, initial_iterable, silent=False,
                 weight_from_key_and_multiplicity=lambda _, multiplicity: multiplicity):
        self.weight_from_key_and_multiplicity = weight_from_key_and_multiplicity
        super().__init__(initial_iterable, silent=silent)

    def __call__(self, iterable, silent=None, weight_from_key_and_multiplicity=None):
        """ Turn an iterable into a numpy array.

                Parameters
                ----------
                iterable : iterable
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

                    >>> vector_from_iterable = VectorFromIterable(['a', 'b', 'c'])

                    Transform an iterable into a vector.

                    >>> assert(set(vector_from_iterable(['b', 'c'])) == {1, 0, 1})
                """

        if not isinstance(iterable, dict):
            if weight_from_key_and_multiplicity is None:
                weight_from_key_and_multiplicity = self.weight_from_key_and_multiplicity
            iterable = {key: weight_from_key_and_multiplicity(key, multiplicity)
                        for key, multiplicity in Counter(iterable).items()}
        return super().__call__(iterable, silent=silent)
