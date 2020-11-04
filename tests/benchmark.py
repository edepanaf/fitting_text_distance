# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import json
import wikipedia
from fitting_text_distance import FittingTextDistance
import time


FILE_WIKIPEDIA_ARTICLES = 'tests/wikipedia_articles.json'


def get_random_wikipedia_article():
    title = wikipedia.random(pages=1)
    try:
        return wikipedia.page(title).content
    except wikipedia.exceptions.DisambiguationError:
        # wikipedia.exceptions.DisambiguationError as e:
        # sometimes, the line wikipedia.page(e.options[0]).content raises again a DisambiguationError exception,
        # which is unfortunate (see for example the wikipedia page 'Shadi')
        return get_random_wikipedia_article()


def save_wikipedia_articles(number_of_articles, file=FILE_WIKIPEDIA_ARTICLES):
    articles = [get_random_wikipedia_article() for _ in range(number_of_articles)]
    open(file, 'w').write(json.dumps(articles))


def load_wikipedia_articles(file=FILE_WIKIPEDIA_ARTICLES):
    return json.loads(open(file, 'r').read())


def benchmark_fitting_distance(number_of_articles, new_articles=False):
    print('Number of articles: ' + str(number_of_articles))
    starting_time = time.time()
    if new_articles:
        save_wikipedia_articles(number_of_articles)
    articles = load_wikipedia_articles()
    load_time = time.time()
    print('Articles downloaded time: ' + str(load_time - starting_time) + ' seconds')
    average_length = sum(len(article) for article in articles) / number_of_articles
    print('Average number of character: ' + str(average_length))
    distance = FittingTextDistance(articles)
    construction_time = time.time()
    print('Distance built time: ' + str(construction_time - load_time) + ' seconds')
    all_distances = dict()
    for i in range(len(articles)):
        for j in range(i+1, len(articles)):
            all_distances[i, j] = distance({articles[i]}, {articles[j]})
    computations_time = time.time()
    print('All distances computed time: ' + str(computations_time - construction_time) + ' seconds')
    print('Number of distances computed: ' + str(number_of_articles * (number_of_articles - 1) / 2))
    return distance, sorted(list(all_distances.items()), key=lambda p: p[1])
