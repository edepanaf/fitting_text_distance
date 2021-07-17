# Fitting text distance

The main class is 'FittingTextDistance'.

Examples
--------

```
>>> from fitting_text_distance.fitting_text_distance import FittingTextDistance
>>> text0 = 'a lovely text'
>>> text1 = 'another lovely text'
>>> text2 = 'something entirely different'
>>> ftd = FittingTextDistance({text0, text1, text2})
```

The 'ftd' object is callable and returns the tf-idf cosine distance between two sets of texts.

```
>>> ftd({text0}, {text0}) # is close to 0.
1.1102230246251565e-16
>>> ftd({text0}, {text1})
0.712889278602614
>>> ftd({text0}, {text2}) # the only common factors appear in all texts, therefore, their tf-idf weight is 0.
1.0
>>> ftd({text0}, {text1, text2})
0.8250313117313128
```

The default tf-idf weights on factors can be adjusted to fit an 'true' distance between two sets of texts.
To represent this true statement, an 'OracleClaim' object is built.
In the following example, this claim is that the distance between '{text0}' and '{text1}'
lies bewteen '0.5' and '0.6'.

```
>>> from fitting_text_distance.oracle_claim import OracleClaim
>>> oracle_claim = OracleClaim(({text0}, {text1}), (0.5, 0.6))
```

We now fit the 'ftd' distance. A set of oracle_claims (here, just one) is provided.

```
>>> ftd.fit({oracle_claim})
```

During this fitting, the factors and texts weights are adjusted.
We can check that the new distance is closer to the claimed one.

```
>>> ftd({text0}, {text1})
0.6129430791488556
```

<!---
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
-->
