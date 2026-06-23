"""Force a headless matplotlib backend for the whole test session.

Imported by pytest before any test module, so the lazy ``import matplotlib.pyplot``
inside ``doe.plotting`` picks up the Agg (non-interactive) backend.
"""

import matplotlib

matplotlib.use("Agg")
