# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


class OracleClaim:
    """ A statement about what the distance between two collections of bags should be.

    Parameters
    ----------
    pair_of_bags: tuple of two collections of bags (immutable iterables)
    distance_interval: tuple of two floats

    Attributes
    ----------
    pair_of_bags: tuple of two collections of bags (immutable iterables)
    distance_interval: tuple of two floats
    """

    def __init__(self, pair_of_bags, distance_interval):
        self.pair_of_bags = pair_of_bags
        self.distance_interval = distance_interval
