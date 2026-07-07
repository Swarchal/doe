# DoE

A Python library for design-of-experiment (DoE) generation and analysis, built on
numpy, scipy, and pandas.

Everything flows through a coded design matrix: *generators* produce a
{class}`~doe.design.Design` (runs stored in natural units, analysed in coded
`[-1, +1]` units), and the *analysis* functions consume one.

```python
import doe

factors = [
    doe.ContinuousFactor("temperature", low=30, high=40, units="C"),
    doe.ContinuousFactor("serum", low=2, high=10, units="%"),
]
design = doe.full_factorial(factors, center=3)
# ... run the experiment, collect a response ...
result = doe.fit_ols(design, response)
```

## Installation

```bash
pip install -e '.[plotting]'   # plotting extra pulls in matplotlib
```

```{toctree}
:maxdepth: 2
:caption: Guides

VIGNETTES
SERIALIZATION
```

```{toctree}
:maxdepth: 1
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

PLAN
PHASE2
PHASE3
PHASE4
UX
```
