# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.fitting_vectorization.fitting_vectorization import FittingVectorization
from fitting_text_distance.distance.cosine_distance import CosineDistance


DEFAULT_SPEED = 0.3
DEFAULT_NUMBER_OF_GRADIENT_STEPS = 6


class FittingDistance:
    """

    Parameters
    ----------
    bag_collection: iterable of immutable iterables
        The immutable iterables are called 'bags'.
    distance: AbstractDistance (optional, default CosineDistance)
    item_to_weight: dict(item: float) (optional, default weight 1. for each item)
    tfidf: bool (optional, default False)
        Whether tf-idf is used to calculate item weights.

    Attributes
    ----------
    vectorize: FittingVectorization
        Callable that turns a collection of bags into a vector (numpy array).
    distance: AbstractDistance

    Methods
    -------
    __call__(bags0: iterable of immutable iterables, bags1: iterable of immutable iterables) -> float
        The distance between the vectorizations of the two bag collections.

    fit(oracle_claims: OracleClaim,
        speed: float between 0. and 1. (optional),
        ratio_item_bag_fitting: float between 0. and 1. (optional, default 0.5),
        number_of_gradient_steps: int (optional)) -> None

    Examples
    --------

    """

    def __init__(self, bag_collection, distance=CosineDistance(), item_to_weight=None, tfidf=False):
        self.vectorize = FittingVectorization(bag_collection, item_to_weight=item_to_weight, tfidf=tfidf)
        self.distance = distance

    def __call__(self, bags0, bags1):
        vectorization0 = self.vectorize(bags0)
        vectorization1 = self.vectorize(bags1)
        return self.distance(vectorization0, vectorization1)

    def fit(self, oracle_claims, speed=DEFAULT_SPEED, ratio_item_bag_fitting=0.5,
            number_of_gradient_steps=DEFAULT_NUMBER_OF_GRADIENT_STEPS):
        tuples_forms_arguments_intervals = [(self.distance, oracle_claim.pair_of_bags, oracle_claim.distance_interval)
                                            for oracle_claim in oracle_claims]
        self.vectorize.fit_from_tuples_of_forms_arguments_intervals(tuples_forms_arguments_intervals,
                                                                    speed=speed,
                                                                    ratio_item_bag_fitting=ratio_item_bag_fitting,
                                                                    number_of_gradient_steps=number_of_gradient_steps)

    def get_bag_weight(self, bag):
        return self.vectorize.get_bag_weight(bag)

    def get_bag_weights(self):
        return self.vectorize.get_bag_weights()

    def get_item_weight(self, item):
        return self.vectorize.get_item_weight(item)

    def get_item_weights(self):
        return self.vectorize.get_item_weights()
