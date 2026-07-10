API reference
=============

The public API is re-exported flat from the top-level :mod:`doe` package
(``doe.full_factorial``, ``doe.fit_ols``, ...); the pages below document it
module by module.

:mod:`doe.factors`
    Factor definitions and natural/coded translation.

:mod:`doe.design`
    The ``Design`` container: runs + factors.

:mod:`doe.generators.factorial`
    Full/fractional factorial and Plackett-Burman generators.

:mod:`doe.generators.rsm`
    Response-surface designs (CCD, Box-Behnken).

:mod:`doe.generators.optimal`
    Computer-generated optimal designs (coordinate exchange, D/I-optimality, augmentation).

:mod:`doe.generators.spacefilling`
    Space-filling designs (Latin hypercube, Sobol, Halton).

:mod:`doe.generators.mixture`
    Mixture designs (simplex-lattice, simplex-centroid, extreme vertices).

:mod:`doe.generators.screening`
    Definitive screening designs (conference-matrix DSDs).

:mod:`doe.analysis.model`
    Model-matrix construction in coded units.

:mod:`doe.analysis.fit`
    OLS fitting and effect estimation.

:mod:`doe.analysis.anova`
    ANOVA, lack-of-fit, and predictive metrics.

:mod:`doe.analysis.diagnostics`
    Design diagnostics: efficiency, VIF, alias structure, leverage, coverage.

:mod:`doe.analysis.optimize`
    Surface optimization: stationary point, constrained optimum, desirability.

:mod:`doe.plotting`
    Effect, RSM, and diagnostic plots (requires the ``plotting`` extra).

:mod:`doe.serialization`
    JSON schema validation for serialized designs.

:mod:`doe.interactive`
    Self-contained HTML run sheets.

.. toctree::
   :maxdepth: 1
   :hidden:

   factors
   design
   generators.factorial
   generators.rsm
   generators.optimal
   generators.spacefilling
   generators.mixture
   generators.screening
   analysis.model
   analysis.fit
   analysis.anova
   analysis.diagnostics
   analysis.optimize
   plotting
   serialization
   interactive
