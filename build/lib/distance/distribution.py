# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


class Distribution:

    def __init__(self, values, probabilities):
        if len(values) != len(probabilities):
            raise ValueError
        self.values = values
        self.probabilities = probabilities
        self.moments = dict()

    def __len__(self):
        return len(self.values)

    def get_moment(self, order):
        return sum(coefficient_wise_vector_product(self.probabilities, power_from_vector(self.values, order)))

    def get_mean(self):
        return self.get_moment(1)

    def get_variance(self):
        return self.get_moment(2) - self.get_moment(1)**2
