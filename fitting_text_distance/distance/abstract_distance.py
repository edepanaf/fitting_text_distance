# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


class AbstractDistance:

    def __init__(self):
        self.partial_gradients = self.first_partial_gradient, self.second_partial_gradient

    def __call__(self, vector0, vector1):
        raise NotImplementedError

    def first_partial_gradient(self, vector0, vector1):
        raise NotImplementedError

    def second_partial_gradient(self, vector0, vector1):
        return self.first_partial_gradient(vector1, vector0)
