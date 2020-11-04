# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import random


def random_float(lower_bound=0., upper_bound=1.):
    return lower_bound + random.random() * upper_bound


def random_vector(length, lower_bound=0., upper_bound=1.):
    return make_vector([random_float(lower_bound=lower_bound, upper_bound=upper_bound) for _ in range(length)])


def random_tuple_from_length(length):
    return tuple(random.random() for _ in range(length))


def probabilities_from_tuple(my_tuple):
    sum_tuple = sum(my_tuple)
    if sum_tuple == 0.:
        raise ValueError
    return tuple(value / sum_tuple for value in my_tuple)


def random_probabilities_tuple_from_length(length):
    if length == 0:
        raise ValueError
    my_tuple = random_tuple_from_length(length)
    while sum(my_tuple) == 0.:
        my_tuple = random_tuple_from_length(length)
    return probabilities_from_tuple(my_tuple)
