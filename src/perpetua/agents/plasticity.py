"""Shared plasticity mechanisms: CReLU activation and scheduled layer grafting.

PPO, DQN/IQL and SAC all expose the same three anti-plasticity-loss options:
a concatenated-ReLU activation, an L2 regularizer (applied in each learner's
own loss) and periodic re-initialization of the deepest ``reset_last_layers``
Dense layers of parameters and optimizer moments. This module holds the two
mechanisms that are independent of any learner's state layout.
"""

# Flax/Optax pytrees are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax  # pyright: ignore[reportMissingTypeStubs]

__all__ = [
    "Params",
    "adam",
    "graft_top_layers",
]

type Params = dict[str, Any]


def crelu(x: jax.Array) -> jax.Array:
    """Concatenated ReLU: ``[relu(x), relu(-x)]`` doubles the feature width."""
    return jnp.concatenate((jax.nn.relu(x), jax.nn.relu(-x)), axis=-1)


def _layer_index(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _top_layer_names(params: Params, count: int, heads: int = 1) -> set[str]:
    """The ``count`` deepest Dense layers, plus any LayerNorm on one of them.

    Flax numbers layers of one type in creation order, which follows depth in
    these networks, so ``Dense_10`` is deeper than ``Dense_9`` (a plain string
    sort gets both of those wrong). A ``LayerNorm_k`` normalises ``Dense_k``.

    ``heads`` is the number of sibling output layers, created last, that read
    the same hidden layer (SAC's mean and log-std heads: 2). They sit at one
    depth and count as one layer, so ``count`` selects the same depth in a
    network with several heads as in one with a single head.
    """
    names = list(cast(dict[str, Any], params["params"]))
    dense = sorted((name for name in names if name.startswith("Dense_")), key=_layer_index)
    if not 0 <= count <= len(dense) - heads + 1:
        raise ValueError("reset_last_layers exceeds the network depth.")
    layers = count + heads - 1 if count else 0
    reset = set(dense[len(dense) - layers :]) if layers else set[str]()
    norms = {name for name in names if name.startswith("LayerNorm_")}
    return reset | {name for name in norms if f"Dense_{_layer_index(name)}" in reset}


def _path_label(step: Any) -> Any:
    """Return a pytree path step's dict key or attribute name when it has one."""
    from jax.tree_util import DictKey, GetAttrKey

    if isinstance(step, DictKey):
        return step.key
    if isinstance(step, GetAttrKey):
        return step.name
    return step


_RESET_GROUP = "reset_layers"
_KEPT_GROUP = "kept_layers"


def _is_reset(path: tuple[Any, ...], reset: set[str]) -> bool:
    """Whether a parameter or optimizer-state leaf belongs to a reset layer.

    A layer counts only as the module directly under ``"params"``, so a
    ``Dense_0`` inside ``CNNTorso_0`` is not the top-level ``Dense_0``. The
    reset group of an :func:`adam` state (its count included) always matches.
    """
    labels = [_path_label(step) for step in path]
    if _RESET_GROUP in labels:
        return True
    if "params" not in labels:
        return False
    module = labels.index("params") + 1
    return module < len(labels) and labels[module] in reset


def adam(reset_last_layers: int | None = None, heads: int = 1) -> optax.GradientTransformation:
    """Adam scaling (``eps=1e-5``), with a separate state for layers that get reset.

    Adam's bias correction depends on its step count. If reset layers shared
    one count with the rest of the network, their zeroed moments would be
    treated as long-run averages and their first steps after a reset would be
    several times the learning rate. With ``reset_last_layers`` set, those
    layers therefore keep their own Adam state, which ``graft_top_layers``
    replaces as a whole. Until the first reset the updates equal plain Adam's.
    ``heads`` is passed to ``_top_layer_names``.
    """
    if not reset_last_layers:
        return optax.scale_by_adam(eps=1e-5)
    count = reset_last_layers

    def labels(params: Params) -> Any:
        reset = _top_layer_names(params, count, heads)

        def label(path: tuple[Any, ...], leaf: Any) -> str:
            del leaf
            return _RESET_GROUP if _is_reset(path, reset) else _KEPT_GROUP

        return jax.tree.map_with_path(label, params)

    return optax.partition(
        {
            _RESET_GROUP: optax.scale_by_adam(eps=1e-5),
            _KEPT_GROUP: optax.scale_by_adam(eps=1e-5),
        },
        labels,
    )


def graft_top_layers(
    params: Params,
    opt_state: optax.OptState,
    fresh_params: Params,
    fresh_opt_state: optax.OptState,
    count: int,
    heads: int = 1,
) -> tuple[Params, optax.OptState]:
    """Replace the deepest ``count`` Dense layers of parameters and optimizer moments.

    Layers are chosen by ``_top_layer_names`` (a LayerNorm on a reset layer is
    reset with it; layers nested in submodules such as ``CNNTorso`` never are;
    ``heads`` sibling output layers count as one).
    Fresh values come from a re-initialized tree with identical structure; all
    other leaves are preserved exactly, and ``count == 0`` keeps everything.
    With an optimizer built by :func:`adam`, the reset layers' Adam step count
    restarts with their moments, so they are bias-corrected like a new network.
    """
    if count == 0:
        return params, opt_state
    reset = _top_layer_names(params, count, heads)

    def graft(path: tuple[Any, ...], new: Any, old: Any) -> Any:
        return new if _is_reset(path, reset) else old

    return (
        cast(Params, jax.tree.map_with_path(graft, fresh_params, params)),
        cast(optax.OptState, jax.tree.map_with_path(graft, fresh_opt_state, opt_state)),
    )
