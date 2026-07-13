"""Cross-cutting wire types: model specs and bounds (``docs/WEBSERVICE_API.md``).

Fields land in Milestone 1 (``docs/WEBSERVICE_BUILD.md`` §1.1).
"""

from typing import Literal

from pydantic import BaseModel

ModelSpecName = Literal["linear", "quadratic", "scheffe-linear", "scheffe-quadratic"]


class ModelSpecObject(BaseModel):
    """``{order: 1 | 2, interactions: bool}``."""

    order: Literal[1, 2]
    interactions: bool = True


#: ``"linear"`` | ``"quadratic"`` | ``"scheffe-linear"`` | ``"scheffe-quadratic"``
#: | ``{order, interactions}`` — accepted everywhere a model is needed
#: (``docs/WEBSERVICE_API.md`` "Model specification").
ModelSpec = ModelSpecName | ModelSpecObject

#: A single coded ``[low, high]`` pair, or a natural-units ``{factor: [low, high]}`` mapping.
Bounds = tuple[float, float] | dict[str, tuple[float, float]]
