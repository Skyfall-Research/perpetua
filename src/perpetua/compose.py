"""Attach a drift process to an environment: ``pp.compose(env, drift)``."""

from collections.abc import Mapping
from typing import Any, cast

from perpetua.core import Environment
from perpetua.drift import Drift
from perpetua.registration import make

__all__ = [
    "compose",
]


def compose[StateT](
    env: Environment[StateT] | str, drift: Drift | Mapping[str, Drift], **kwargs: Any
) -> Environment[StateT]:
    """Return an environment equivalent to ``env`` but driven by ``drift``.

    Accepts a registered environment id (extra keyword arguments are forwarded
    to ``make``) or an already-constructed environment that supports
    ``with_drift``. Environments whose drift cannot be rebound after
    construction raise a clear error instead of being silently copied, so a
    composed lifetime never runs a different drift than the caller asked for.
    Compose before wrapping: an environment already wrapped in
    :class:`~perpetua.Oracle` or :class:`~perpetua.OneHotObservation` raises
    ``TypeError`` naming the wrapper. A bare process drives the environment's
    ``default_drift_field`` and raises ``ValueError``, as the constructor does,
    when there is no unambiguous field; pass a mapping from parameter names to
    processes in that case.
    """
    if isinstance(env, str):
        return cast(Environment[StateT], cast(Any, make)(env, drift=drift, **kwargs))
    if kwargs:
        raise ValueError("Keyword arguments are only valid with a registered environment id.")
    rebind = getattr(env, "with_drift", None)
    if rebind is None:
        raise TypeError(
            f"{type(env).__name__} does not support rebinding drift after construction; "
            "pass drift= to make() instead."
        )
    return cast(Environment[StateT], rebind(drift))
