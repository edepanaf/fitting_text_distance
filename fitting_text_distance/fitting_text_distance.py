# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_text_distance.fitting_distance import FittingDistance
from fitting_text_distance.distance.cosine_distance import CosineDistance
from fitting_text_distance.oracle_claim import OracleClaim


DEFAULT_MAX_FACTOR_LENGTH = 5
DEFAULT_ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789'


class FittingTextDistance:
    """ A distance on text collections able to learn from examples.

        Parameters
        ----------
        text_collection : iterable of str
            The collection of all texts on which the distance will be defined.
        distance (optional) : Distance
            Can be either CosineDistance() (by default) or JensenShannonDistance()
        max_factor_length (optional) : int
            Maximum length of factors considered. By default is set to 5.
        factor_to_weight (optional) : dict[str -> float]
            The weight associated to each factor. By default, the tf-idf weights are used.
        alphabet (optional) : iterable
            The set of letters that are kept in the texts. The other letters are replaced by a space ' '.
            By default, equal to 'abcdefghijklmnopqrstuvwxyz0123456789'.

        Attributes
        ----------
        text_to_bag : dict
            links each text to its bag of factors
        fitting_distance : FittingDistance
            distance on bags of factors

        Examples
        --------
        >>> from fitting_text_distance.fitting_text_distance import FittingTextDistance
        >>> text0 = 'a lovely text'
        >>> text1 = 'another lovely text'
        >>> text2 = 'something entirely different'
        >>> ftd = FittingTextDistance({text0, text1, text2})

        The 'ftd' object is callable and returns the tf-idf cosine distance between two sets of texts.

        >>> ftd({text0}, {text0}) # is close to 0.
        1.1102230246251565e-16
        >>> ftd({text0}, {text1})
        0.712889278602614
        >>> ftd({text0}, {text2}) # the only common factors appear in all texts, therefore, their tf-idf weight is 0.
        1.0
        >>> ftd({text0}, {text1, text2})
        0.8250313117313128

        The default tf-idf weights on factors can be adjusted to fit an 'true' distance between two sets of texts.
        To represent this true statement, an 'OracleClaim' object is built.
        In the following example, this claim is that the distance between '{text0}' and '{text1}'
        lies bewteen '0.5' and '0.6'.

        >>> from fitting_text_distance.oracle_claim import OracleClaim
        >>> oracle_claim = OracleClaim(({text0}, {text1}), (0.5, 0.6))

        We now fit the 'ftd' distance. A set of oracle_claims (here, just one) is provided.

        >>> ftd.fit({oracle_claim})

        During this fitting, the factors and texts weights are adjusted.
        We can check that the new distance is closer to the claimed one.

        >>> ftd({text0}, {text1})
        0.6129430791488556

        """

    def __init__(self, text_collection, distance=CosineDistance(),
                 max_factor_length=DEFAULT_MAX_FACTOR_LENGTH,
                 factor_to_weight=None,
                 alphabet=DEFAULT_ALPHABET):
        self.text_to_bag = {text: bag_of_factors_from_text(clean_text(text, alphabet=alphabet), max_factor_length)
                            for text in text_collection}
        if isinstance(text_collection, dict):
            bags = {self.text_to_bag[text]: weight for text, weight in text_collection.items()}
        else:
            bags = self.text_to_bag.values()
        self.fitting_distance = FittingDistance(bags, distance, item_to_weight=factor_to_weight, tfidf=True)

    def __call__(self, text_collection0, text_collection1):
        bags0 = self.bag_collection_from_text_collection(text_collection0)
        bags1 = self.bag_collection_from_text_collection(text_collection1)
        return self.fitting_distance(bags0, bags1)

    def fit(self, text_oracle_claims):
        bag_oracle_claims = {self.bag_oracle_claim_from_text_oracle_claim(claim) for claim in text_oracle_claims}
        self.fitting_distance.fit(bag_oracle_claims)

    def get_factor_weight(self, factor):
        try:
            return self.fitting_distance.get_item_weight(factor)
        except ValueError:
            return 0.

    def get_text_weight(self, text):
        return self.fitting_distance.get_bag_weight(self.text_to_bag[text])

    def bag_collection_from_text_collection(self, text_collection):
        return {self.text_to_bag[text] for text in text_collection}

    def bag_oracle_claim_from_text_oracle_claim(self, text_oracle_claim):
        text_collection0, text_collection1 = text_oracle_claim.pair_of_bags
        bags0 = self.bag_collection_from_text_collection(text_collection0)
        bags1 = self.bag_collection_from_text_collection(text_collection1)
        return OracleClaim((bags0, bags1), text_oracle_claim.distance_interval)


def bag_of_factors_from_text(text, max_factor_length=None):
    if max_factor_length is None:
        max_factor_length = len(text)
    return tuple(text[start:end] for start in range(len(text))
                 for end in range(start + 1, 1 + min(len(text), start + max_factor_length)))


def clean_text(text, alphabet=DEFAULT_ALPHABET):
    return ''.join(clean_letter(letter, alphabet=alphabet) for letter in text)


def clean_letter(letter, alphabet=DEFAULT_ALPHABET):
    space = ' '
    lower_letter = letter.lower()
    if lower_letter in alphabet:
        return lower_letter
    return space
