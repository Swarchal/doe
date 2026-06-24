# Known issues

No major open issues are currently tracked here.

The previously listed issues were fixed:

- `plackett_burman` and `fractional_factorial` now reject categorical factors with more
  than two levels instead of silently dropping middle levels.
- `Design.randomize()` can be called repeatedly while preserving a single `std_order`
  column that tracks the original standard order.
