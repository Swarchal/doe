# End-to-end workflow: from factors to an operating point

This walkthrough shows a typical DoE loop:

1. define the factors and practical ranges,
2. choose a design that can estimate the model you need,
3. randomize and run the experiment,
4. attach the response,
5. fit and check the model,
6. use the fitted surface to choose the next setting.

The example is a small reaction-optimization study with three continuous factors:
temperature, reaction time, and catalyst loading. The response is percent yield.
The same pattern applies to assays, formulations, process screens, and other
continuous-input experiments.

## 1. Define the experimental space

Start with factor ranges that are safe and practically meaningful. DoE fits in coded
units internally, but the design is stored and displayed in natural units.

```python
from doe import ContinuousFactor

factors = [
    ContinuousFactor("temperature", low=45, high=75, units="C"),
    ContinuousFactor("time", low=20, high=60, units="min"),
    ContinuousFactor("catalyst", low=0.5, high=2.5, units="g/L"),
]
```

The coded midpoint is zero for every continuous factor, so the center of this region is
60 C, 40 min, and 1.5 g/L.

## 2. Generate and randomize the design

If the goal is optimization, use a response-surface design rather than a two-level
screening design. A central composite design has enough levels to estimate curvature
and interactions.

```python
from doe import central_composite

design = central_composite(factors, alpha="faced", center=5).randomize(seed=20260707)

print(design.n_runs, design.n_center)
print(design.runs.head(8).round(2))
```

```text
19 5
   std_order  temperature  time  catalyst
0         12         60.0  40.0       0.5
1         16         60.0  40.0       1.5
2          1         45.0  20.0       2.5
3          8         45.0  40.0       1.5
4         17         60.0  40.0       1.5
5         14         60.0  40.0       1.5
6          6         75.0  60.0       0.5
7         13         60.0  40.0       2.5
```

The `std_order` column records the original design order before randomization. Keep the
randomized order for execution: it protects the analysis from time trends such as a
warming instrument, ageing reagents, or operator fatigue.

## 3. Add the measured response

After running the experiment, attach the measured response to the design. Keeping the
response on the design prevents accidental misalignment between run order and response
values.

```python
import numpy as np

# Replace this block with the response values measured in the lab or plant.
coded = design.coded()
rng = np.random.default_rng(42)
yield_pct = (
    78
    + 7.5 * coded["temperature"]
    + 5.0 * coded["time"]
    + 3.0 * coded["catalyst"]
    - 8.0 * coded["temperature"] ** 2
    - 5.5 * coded["time"] ** 2
    - 4.0 * coded["catalyst"] ** 2
    + 2.5 * coded["temperature"] * coded["time"]
    - 1.5 * coded["time"] * coded["catalyst"]
    + rng.normal(0, 0.8, design.n_runs)
)

measured = design.with_response("yield_pct", yield_pct)
```

Real analyses should use the measured values in randomized run order. The synthetic
response above is only there to make the walkthrough reproducible.

## 4. Fit a quadratic model

A central composite design is intended for a second-order response-surface model:
main effects, two-factor interactions, and squared terms.

```python
import pandas as pd
from doe import fit_ols

fit = fit_ols(measured, "yield_pct", model="quadratic")

print(f"R2={fit.r_squared:.3f}")
print(f"adjusted R2={fit.adjusted_r2():.3f}")
print(f"predicted R2={fit.predicted_r2():.3f}")

summary = pd.DataFrame(fit.summary(), index=["coefficient", "effect"]).T
print(summary.round(2))
```

```text
R2=0.998
adjusted R2=0.995
predicted R2=0.985
                      coefficient  effect
Intercept                   77.50   77.50
temperature                  7.39   14.78
time                         5.03   10.05
catalyst                     2.93    5.86
temperature:time             2.53    5.06
temperature:catalyst        -0.22   -0.45
time:catalyst               -1.26   -2.53
temperature^2               -7.24  -14.47
time^2                      -5.61  -11.21
catalyst^2                  -3.76   -7.53
```

In coded units, an effect is the fitted low-to-high swing. Here, raising temperature
from 45 C to 75 C is associated with about a 14.8 percentage-point yield increase near
the center of the design, before accounting for curvature and interactions.

## 5. Check whether the model is usable

Look for three things before acting on the model:

- high adjusted and predicted R2 values,
- no strong lack-of-fit signal,
- no serious collinearity in the fitted terms.

```python
from doe import vif

print(fit.anova().round(3))

lof = fit.lack_of_fit()
print(f"lack-of-fit p={lof.p_value:.3f}")

print(vif(fit.model_matrix, term_names=fit.term_names).round(2))
```

```text
                           SS   df       MS         F      p
temperature           546.317  1.0  546.317  1047.064  0.000
time                  252.662  1.0  252.662   484.249  0.000
catalyst               85.914  1.0   85.914   164.661  0.000
temperature:time       51.250  1.0   51.250    98.224  0.000
temperature:catalyst    0.398  1.0    0.398     0.762  0.405
time:catalyst          12.774  1.0   12.774    24.482  0.001
temperature^2         758.010  1.0  758.010  1452.792  0.000
time^2                153.987  1.0  153.987   295.130  0.000
catalyst^2             38.710  1.0   38.710    74.192  0.000
Residual                4.696  9.0    0.522       NaN    NaN
Total                1904.717 18.0      NaN       NaN    NaN

lack-of-fit p=0.755

temperature             1.00
time                    1.00
catalyst                1.00
temperature:time        1.00
temperature:catalyst    1.00
time:catalyst           1.00
temperature^2           1.73
time^2                  1.73
catalyst^2              1.73
Name: VIF, dtype: float64
```

The large lack-of-fit p-value means the residual variation is consistent with pure
experimental error from replicated settings. The VIF values are small, so the model
terms are not fighting severe collinearity.

## 6. Choose the operating point

With a quadratic model, the fitted surface can be optimized directly. `stationary_point`
reports the unconstrained stationary point and classifies it; `optimum` searches within
the coded design box.

```python
stationary = fit.stationary_point()
optimum = fit.optimum(maximize=True)

print(stationary)
print(optimum)
```

```text
StationaryPoint(maximum: temperature=69.05, time=51.06, catalyst=1.779 -> 81.53)
Optimum(max: temperature=69.05, time=51.06, catalyst=1.779 -> 81.53)
```

In this case the stationary point is a feasible maximum, so the unconstrained and
box-constrained optima agree.

## 7. Plan the confirmation run

The model suggests a confirmation setting in natural units:

```python
confirmation = pd.DataFrame([optimum.natural]).round(2)
print(confirmation)
print(fit.predict(optimum.natural).round(2))
```

```text
   temperature   time  catalyst
0        69.05  51.06      1.78
[81.53]
```

Run one or more confirmation experiments near this point. If the observed response
matches the prediction within the experimental error seen in the design, the workflow
has produced an operating point. If it does not, augment the design around the region
where the model failed and refit before making a process decision.

