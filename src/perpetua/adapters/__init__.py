"""Explicit compatibility views and host-backed continuing lifetimes."""

from typing import TYPE_CHECKING, Any

from perpetua.adapters.gymnasium import GymnasiumAdapter
from perpetua.adapters.host import EnvPoolCartPole, GymnasiumLifetime

if TYPE_CHECKING:
    from perpetua.adapters.pettingzoo import (
        AECLifetime,
        PettingZooAdapter,
        PettingZooLifetime,
    )

__all__ = [
    "AECLifetime",
    "EnvPoolCartPole",
    "GymnasiumAdapter",
    "GymnasiumLifetime",
    "PettingZooAdapter",
    "PettingZooLifetime",
]


def __getattr__(name: str) -> Any:
    if name in {"AECLifetime", "PettingZooAdapter", "PettingZooLifetime"}:
        from perpetua.adapters.pettingzoo import (
            AECLifetime,
            PettingZooAdapter,
            PettingZooLifetime,
        )

        return {
            "AECLifetime": AECLifetime,
            "PettingZooAdapter": PettingZooAdapter,
            "PettingZooLifetime": PettingZooLifetime,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
