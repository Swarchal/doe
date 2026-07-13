"""Factor schemas — the discriminated union mirroring ``doe.factor_from_dict``.

Fields land in Milestone 1 (``docs/WEBSERVICE_BUILD.md`` §1.1), field-for-field with
the serialization schema (``docs/SERIALIZATION.md``) and ``doe.factors.factor_from_dict``.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from doe import CategoricalFactor, ContinuousFactor, Factor, MixtureFactor


class _HardToChangeMixin(BaseModel):
    """The optional split-plot ``hard_to_change`` flag (``docs/SERIALIZATION.md``).

    Emitted **only when true**, exactly as ``Factor.to_dict`` does it, so a pre-split-plot
    factor serializes byte-for-byte as before -- the existing generation contracts
    (``central-composite``, ``optimal``, ``augment``) embed factor documents and compare
    key sets exactly.
    """

    hard_to_change: bool = False

    @model_serializer(mode="wrap")
    def _drop_default_hard_to_change(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if not self.hard_to_change:
            data.pop("hard_to_change", None)
        return data


class ContinuousFactorSchema(_HardToChangeMixin):
    """``{type: "continuous", name, low, high, units?, hard_to_change?}``."""

    type: Literal["continuous"] = "continuous"
    name: str
    low: float
    high: float
    units: str | None = None

    def to_factor(self) -> ContinuousFactor:
        return ContinuousFactor(
            name=self.name,
            low=self.low,
            high=self.high,
            units=self.units,
            hard_to_change=self.hard_to_change,
        )


class CategoricalFactorSchema(_HardToChangeMixin):
    """``{type: "categorical", name, levels, units?, hard_to_change?}``."""

    type: Literal["categorical"] = "categorical"
    name: str
    levels: list[str]
    units: str | None = None

    def to_factor(self) -> CategoricalFactor:
        return CategoricalFactor(
            name=self.name,
            levels=tuple(self.levels),
            units=self.units,
            hard_to_change=self.hard_to_change,
        )


class MixtureFactorSchema(BaseModel):
    """``{type: "mixture", name, low?, high?, units?}`` — proportion bounds in ``[0, 1]``."""

    type: Literal["mixture"] = "mixture"
    name: str
    low: float = 0.0
    high: float = 1.0
    units: str | None = None

    def to_factor(self) -> MixtureFactor:
        return MixtureFactor(name=self.name, low=self.low, high=self.high, units=self.units)


FactorSchema = Annotated[
    ContinuousFactorSchema | CategoricalFactorSchema | MixtureFactorSchema,
    Field(discriminator="type"),
]


def factor_schema_to_factor(
    schema: ContinuousFactorSchema | CategoricalFactorSchema | MixtureFactorSchema,
) -> Factor:
    """Convert any factor schema variant to its real ``doe`` factor object."""
    return schema.to_factor()
