# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from collections import Counter
from fitting_text_distance.fitting_vectorization import VectorFromDict


class VectorFromIterable(VectorFromDict):

    def __init__(self, initial_iterable, silent=False,
                 weight_from_key_and_multiplicity=lambda _, multiplicity: multiplicity):
        self.weight_from_key_and_multiplicity = weight_from_key_and_multiplicity
        super().__init__(initial_iterable, silent=silent)

    def __call__(self, iterable, silent=None, weight_from_key_and_multiplicity=None):
        if not isinstance(iterable, dict):
            if weight_from_key_and_multiplicity is None:
                weight_from_key_and_multiplicity = self.weight_from_key_and_multiplicity
            iterable = {key: weight_from_key_and_multiplicity(key, multiplicity)
                        for key, multiplicity in Counter(iterable).items()}
        return super().__call__(iterable, silent=silent)
