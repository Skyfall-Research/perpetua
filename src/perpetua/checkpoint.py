"""Atomic Orbax checkpoints for full lifetime/training pytrees and configuration."""

# Orbax and JAX expose some partially annotated helpers.
# pyright: reportUnknownMemberType=false

import importlib.metadata
import json
import shutil
import sys
import uuid
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp  # pyright: ignore[reportMissingTypeStubs]

from perpetua import reporting
from perpetua.core import JSONValue, as_int
from perpetua.runtime import ChunkRunner

__all__ = [
    "save",
    "restore",
    "provenance",
    "replay",
]


def _configuration(metadata: Mapping[str, object]) -> dict[str, JSONValue]:
    """Normalize JSON-compatible metadata; reject objects and non-finite numbers."""
    return cast(
        dict[str, JSONValue],
        json.loads(json.dumps(dict(metadata), allow_nan=False, sort_keys=True)),
    )


def _refuses_checkpoints(entry: Mapping[str, JSONValue]) -> bool:
    # Same default as runtime and testing: a host simulator's state lives
    # outside the JAX carry, so it is checkpointable only when it says so.
    declared = entry.get("checkpointable")
    return declared is False or (declared is None and entry.get("execution") == "host")


def _require_checkpoint_support(configuration: dict[str, JSONValue]) -> None:
    environment = configuration.get("environment")
    if _refuses_checkpoints(configuration) or (
        isinstance(environment, dict) and _refuses_checkpoints(environment)
    ):
        raise NotImplementedError("This backend does not expose a complete checkpointable state.")


def _require_complete_carry(*objects: object) -> None:
    # Looked up rather than imported: the bridge needs the optional torch extra,
    # and neither class can exist before the module has been imported.
    bridge = sys.modules.get("perpetua.torch_bridge")
    if bridge is None:
        return
    external = (bridge.TorchRunner, bridge.TorchTrainingState)
    if any(isinstance(item, external) for item in objects):
        raise TypeError(
            "A TorchRunner carry does not hold the agent: its networks, optimizer and sampling "
            "state live in the PyTorch agent object, so this checkpoint would not contain them "
            "and a continuation from it would not repeat the run. Use TorchRunner.save and "
            "TorchRunner.restore, which write and load both together."
        )


_VERSION = 2


def _nodes(tree: object, path: str = "") -> list[JSONValue]:
    """``[path, type name]`` for every container node, depth first.

    The description depends only on key paths and node type names, never on
    ``str(treedef)``: that prints a node's static (aux) data, which for an
    object with the default repr contains a memory address that differs in
    every process.
    """

    def is_child(node: object) -> bool:
        return node is not tree

    children = cast(
        "list[tuple[tuple[object, ...], object]]",
        jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_child)[0],
    )
    if len(children) == 1 and children[0][0] == ():
        return []
    found: list[JSONValue] = [[path, type(tree).__qualname__]]
    for key, child in children:
        found.extend(_nodes(child, path + jax.tree_util.keystr(key)))
    return found


def _schema(state: object) -> dict[str, JSONValue]:
    """Record structure, shapes and dtypes independently of checkpoint values."""
    leaves = cast(
        "list[tuple[tuple[object, ...], object]]",
        jax.tree_util.tree_flatten_with_path(state)[0],
    )
    specs: list[JSONValue] = []
    for key, value in leaves:
        if isinstance(value, jax.Array):
            shape = value.shape
            dtype = str(value.dtype)
        elif isinstance(value, np.ndarray):
            array = cast(np.ndarray[tuple[int, ...], np.dtype[Any]], value)
            shape, dtype = array.shape, str(array.dtype)
        elif isinstance(value, (int, float, bool)):
            array = np.asarray(value)
            shape, dtype = array.shape, str(array.dtype)
        else:
            raise TypeError("Checkpoint leaves must be numeric arrays or scalars.")
        specs.append({"path": jax.tree_util.keystr(key), "shape": list(shape), "dtype": dtype})
    return {"nodes": _nodes(state), "leaves": specs}


def _show(value: object) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 80 else text[:77] + "..."


def _configuration_differences(saved: Any, expected: Any, path: str = "metadata") -> list[str]:
    """Dotted paths that differ between two JSON trees, with both values."""
    if isinstance(saved, dict) and isinstance(expected, dict):
        old, new = cast(dict[str, Any], saved), cast(dict[str, Any], expected)
        found: list[str] = []
        for key in sorted(set(old) | set(new)):
            where = f"{path}.{key}"
            if key not in old:
                found.append(f"{where}: not in the checkpoint, now {_show(new[key])}")
            elif key not in new:
                found.append(f"{where}: {_show(old[key])} in the checkpoint, now absent")
            else:
                found.extend(_configuration_differences(old[key], new[key], where))
        return found
    if isinstance(saved, list) and isinstance(expected, list):
        olds, news = cast(list[Any], saved), cast(list[Any], expected)
        if len(olds) == len(news) and any(isinstance(item, (dict, list)) for item in olds):
            found = []
            for index, (old_item, new_item) in enumerate(zip(olds, news, strict=True)):
                found.extend(_configuration_differences(old_item, new_item, f"{path}[{index}]"))
            return found
    if saved == expected:
        return []
    was, now = cast(object, saved), cast(object, expected)
    return [f"{path}: {_show(was)} in the checkpoint, {_show(now)} expected"]


def _schema_differences(saved: Any, expected: dict[str, JSONValue]) -> list[str]:
    """What differs between a saved schema and the template's, most structural first."""
    if not isinstance(saved, dict):
        return ["the checkpoint's schema is unreadable"]
    record = cast(dict[str, Any], saved)
    found: list[str] = []
    old_nodes = cast(list[list[str]], record.get("nodes", []))
    new_nodes = cast(list[list[str]], expected["nodes"])
    for (old_path, old_type), (new_path, new_type) in zip(old_nodes, new_nodes, strict=False):
        if old_path != new_path:
            found.append(
                f"container {old_path or '<root>'!r} ({old_type}) in the checkpoint where the "
                f"template has {new_path or '<root>'!r} ({new_type})"
            )
            break
        if old_type != new_type:
            found.append(
                f"container at {new_path or '<root>'!r}: {old_type} in the checkpoint, "
                f"{new_type} in the template"
            )
            break
    else:
        if len(old_nodes) != len(new_nodes):
            found.append(
                f"{len(old_nodes)} containers in the checkpoint, {len(new_nodes)} in the template"
            )
    old_leaves = cast(list[dict[str, Any]], record.get("leaves", []))
    new_leaves = cast(list[dict[str, Any]], expected["leaves"])
    for old, new in zip(old_leaves, new_leaves, strict=False):
        if old.get("path") != new["path"]:
            found.append(
                f"leaf {old.get('path')!r} in the checkpoint, {new['path']!r} in the template"
            )
            break
        if old.get("shape") != new["shape"] or old.get("dtype") != new["dtype"]:
            found.append(
                f"leaf {new['path'] or '<root>'!r}: {old.get('dtype')}{old.get('shape')} in the "
                f"checkpoint, {new['dtype']}{new['shape']} in the template"
            )
            break
    else:
        if len(old_leaves) != len(new_leaves):
            found.append(
                f"{len(old_leaves)} array leaves in the checkpoint, "
                f"{len(new_leaves)} in the template"
            )
    return found or ["the schemas differ"]


def _software() -> dict[str, JSONValue]:
    return {**reporting.provenance(), "orbax": importlib.metadata.version("orbax-checkpoint")}


def _software_differences(saved: object) -> list[str]:
    """Why the recorded software is not known to be the running one.

    An entry that differs is listed; one that either side does not know is
    skipped, except the revision. A checkpoint that records a revision entry
    is also listed when only one side knows its revision and when either side
    had uncommitted changes, since neither then identifies the code.
    """
    if not isinstance(saved, dict):
        return []
    record, running = cast(dict[str, Any], saved), _software()
    found = [
        f"{name} {record[name]} in the checkpoint, {running[name]} now"
        for name in ("perpetua", "revision", "jax", "orbax", "backend")
        if record.get(name) is not None
        and running[name] is not None
        and record[name] != running[name]
    ]
    if "revision" not in record:
        return found
    was, now = record["revision"], running["revision"]
    if was is None and now is not None:
        found.append(f"revision unknown in the checkpoint, {now} now")
    elif was is not None and now is None:
        found.append(f"revision {was} in the checkpoint, unknown now")
    if record.get("dirty") is True:
        found.append("the checkpoint was written from a revision with uncommitted changes")
    if running["dirty"] is True:
        found.append("the running revision has uncommitted changes")
    return found


def _payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}: the path does not exist.")
    missing = [item for item in ("state", "metadata") if not (path / item).is_dir()]
    if not path.is_dir() or missing:
        raise ValueError(
            f"{path} is not a Perpetua checkpoint: missing {', '.join(missing) or 'its items'}."
        )
    with cast(Any, ocp.Checkpointer(ocp.CompositeCheckpointHandler())) as checkpointer:
        header = checkpointer.restore(
            path, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())
        )
    payload = header.metadata
    if not isinstance(payload, dict) or payload.get("format") != "perpetua-lifetime":
        raise ValueError(f"{path} is not a Perpetua lifetime checkpoint.")
    return cast(dict[str, Any], payload)


def _is_empty(leaf: object) -> bool:
    return isinstance(leaf, (jax.Array, np.ndarray)) and cast(Any, leaf).size == 0


def _placeholders(tree: object) -> object:
    """Replace zero-size array leaves, which Orbax cannot save, with one element.

    An empty array carries no data and the schema already records its shape and
    dtype, so restore takes the empty leaf back from the template.
    """

    def fill(leaf: Any) -> Any:
        return jnp.zeros((1,), dtype=leaf.dtype) if _is_empty(leaf) else leaf

    return jax.tree.map(fill, tree)


def _is_array_record(node: object) -> bool:
    return hasattr(node, "dtype") and hasattr(node, "shape")


def _stored_differences(path: Path, template: object) -> list[str]:
    """Stored arrays whose shape or dtype is not the template's.

    The schema in the header is a JSON record beside the arrays. This reads
    Orbax's own description of each stored array, without loading any, so a
    state directory that does not belong to its header is refused instead of
    being cast into the template.
    """
    record = cast(Any, ocp.StandardCheckpointHandler()).metadata(path / "state")
    records = cast(
        "list[tuple[tuple[object, ...], Any]]",
        jax.tree_util.tree_flatten_with_path(record.tree, is_leaf=_is_array_record)[0],
    )
    stored = {
        jax.tree_util.keystr(key): (tuple(node.shape), str(np.dtype(node.dtype)))
        for key, node in records
    }
    names = cast(Any, ocp.utils).serialize_tree(_placeholders(template), keep_empty_nodes=True)
    leaves = cast(
        "list[tuple[tuple[object, ...], Any]]", jax.tree_util.tree_flatten_with_path(names)[0]
    )
    found: list[str] = []
    for key, leaf in leaves:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        value = np.asarray(leaf)
        where, expected = jax.tree_util.keystr(key), (tuple(value.shape), str(value.dtype))
        kept = stored.pop(where, None)
        if kept is None:
            found.append(f"no stored array for leaf {where!r}")
        elif kept != expected:
            found.append(
                f"leaf {where!r}: {kept[1]}{list(kept[0])} stored in the checkpoint, "
                f"{expected[1]}{list(expected[0])} in the template"
            )
    return found + [f"stored array {where!r} is not in the template" for where in stored]


def save(path: str | Path, state: object, *, metadata: Mapping[str, object]) -> None:
    """Synchronously save a complete, versioned checkpoint without overwriting.

    Orbax commits state and metadata atomically inside a private staging
    directory, which is renamed into place only after every write succeeds. A
    process killed mid-save therefore leaves no checkpoint at ``path`` to
    restore or to block a retry. All JAX keys, including typed PRNG keys, are
    handled by Orbax's standard array serialization. Save only at returned
    runner boundaries, not during a compiled update.

    Raises ``NotImplementedError`` when ``metadata`` (or its ``environment``
    entry) declares ``checkpointable`` False, or is a host simulator
    (``execution`` ``"host"``) that does not declare it, ``TypeError`` for a state leaf
    that is not a numeric array or scalar and ``FileExistsError`` when
    ``path`` exists. Metadata must be JSON-serialisable with finite numbers
    (``TypeError`` or ``ValueError`` otherwise).

    The state must be the complete carry. A
    :class:`perpetua.torch_bridge.TorchTrainingState` is refused with
    ``TypeError``, here and as a ``restore`` template: the PyTorch agent's
    state is not in it, and :meth:`perpetua.torch_bridge.TorchRunner.save`
    writes both.

    The checkpoint also records what wrote it: the entries of
    :func:`perpetua.reporting.provenance` (Perpetua version, source revision,
    JAX version, backend, Python, platform) and the Orbax version. Read them
    back with :func:`provenance`.
    """
    path = Path(path).expanduser().resolve()
    _require_complete_carry(state)
    configuration = _configuration(metadata)
    _require_checkpoint_support(configuration)
    schema = _schema(state)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload: dict[str, JSONValue] = {
        "format": "perpetua-lifetime",
        "version": _VERSION,
        "configuration": configuration,
        "schema": schema,
        "software": _software(),
    }
    try:
        # Keep the untyped third-party interface at this serialization boundary.
        with cast(Any, ocp.Checkpointer(ocp.CompositeCheckpointHandler())) as checkpointer:
            checkpointer.save(
                staging,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(_placeholders(state)),
                    metadata=ocp.args.JsonSave(payload),
                ),
            )
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def restore[State](
    path: str | Path, template: State, *, expected_metadata: Mapping[str, object]
) -> State:
    """Restore into a matching template after validating configuration and schema.

    A template supplies types and target shardings, not initialization values.
    No model code is imported from the checkpoint. Deterministic continuation
    is validated on matching software/backends, not promised across devices.

    Raises ``NotImplementedError`` when ``expected_metadata`` declares
    ``checkpointable`` False (or is an undeclared host simulator, as for
    ``save``), ``FileNotFoundError`` when ``path`` does not exist, and
    ``ValueError`` when the checkpoint is missing its items, has an
    unsupported or older format version, or its configuration or schema does
    not match ``expected_metadata`` and ``template``. A mismatch error names
    the first differing configuration keys with both values, or the first
    differing container or array leaf (path, dtype and shape). The shapes and
    dtypes of the arrays actually stored are compared with the template too,
    from Orbax's record of them and before any array is loaded, so stored
    arrays that disagree with the checkpoint's own header are refused rather
    than cast.

    The saved structure is the key path of every array leaf plus the type
    name of every container, not JAX's ``treedef`` string, so a checkpoint
    restores in a new process even when a pytree node carries static fields.
    The static values themselves are not compared; record them in the
    metadata if they matter.

    The software the checkpoint was written with is compared with the running
    one. A different Perpetua version or source revision, JAX or Orbax version
    or backend does not refuse the restore: it emits one ``UserWarning``
    listing the differences, since exact continuation is only validated on
    matching software. The same warning also lists a revision that only one
    side knows, and uncommitted changes on either side (``dirty`` True), since
    a dirty or unknown revision does not identify the code; restoring in a
    checkout with uncommitted changes therefore always warns. Other entries
    that either side does not know are skipped, as is the revision of a
    checkpoint written before revisions were recorded.
    """
    path = Path(path).expanduser().resolve()
    _require_complete_carry(template)
    expected = _configuration(expected_metadata)
    _require_checkpoint_support(expected)
    payload = _payload(path)
    with cast(Any, ocp.Checkpointer(ocp.CompositeCheckpointHandler())) as checkpointer:
        version = cast(Any, payload.get("version"))
        if isinstance(version, int) and version < _VERSION:
            raise ValueError(
                f"Checkpoint format version {version} is older than the supported version "
                f"{_VERSION} (its structure record was not process-independent) and cannot be "
                "restored; re-run to create a new checkpoint."
            )
        if version != _VERSION:
            raise ValueError(
                f"Unsupported Perpetua checkpoint format version {version!r}; this release "
                f"reads version {_VERSION}."
            )
        saved_configuration = cast(Any, payload.get("configuration"))
        if saved_configuration != expected:
            differences = _configuration_differences(saved_configuration, expected)
            more = len(differences) - 3
            raise ValueError(
                "Checkpoint configuration does not match this experiment: "
                + "; ".join(differences[:3])
                + (f"; and {more} more" if more > 0 else "")
                + "."
            )
        schema = _schema(template)
        if payload.get("schema") != schema:
            differences = _schema_differences(cast(Any, payload.get("schema")), schema)[:3]
            text = "; ".join(differences)
            hint = (
                " 64-bit types appear in the difference: jax_enable_x64 must be set the same way "
                "when saving and restoring."
                if any(name in text for name in ("float64", "int64"))
                else ""
            )
            raise ValueError(
                f"Checkpoint structure, shapes or dtypes do not match the template: {text}.{hint}"
            )
        mixed = _stored_differences(path, template)
        if mixed:
            raise ValueError(
                "Checkpoint structure, shapes or dtypes do not match the template: "
                + "; ".join(mixed[:3])
                + ". The stored arrays do not belong to this checkpoint's header."
            )
        changed = _software_differences(payload.get("software"))
        if changed:
            warnings.warn(
                "This checkpoint is not known to come from the running software: "
                + "; ".join(changed)
                + ". Exact continuation is validated on matching software and backends only.",
                stacklevel=2,
            )
        restored = checkpointer.restore(
            path,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(_placeholders(template), strict=True),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        def unfill(original: Any, loaded: Any) -> Any:
            return original if _is_empty(original) else loaded

        return cast(State, jax.tree.map(unfill, template, restored.state))


def provenance(path: str | Path) -> dict[str, JSONValue]:
    """What wrote the checkpoint at ``path``: its recorded software entries.

    The keys of :func:`perpetua.reporting.provenance` plus ``orbax``, as they
    were when the checkpoint was saved. A checkpoint from before the revision
    was recorded has only ``jax``, ``orbax`` and ``backend``. Reads the
    checkpoint's JSON header only, so it needs no template and works for any
    format version. Raises ``FileNotFoundError`` or ``ValueError`` as
    :func:`restore` does for a missing path or a directory that is not a
    checkpoint.
    """
    software = _payload(Path(path).expanduser().resolve()).get("software")
    return cast(dict[str, JSONValue], software) if isinstance(software, dict) else {}


def _copy_leaf(leaf: Any) -> Any:
    return jnp.copy(leaf) if isinstance(leaf, jax.Array) else leaf


def replay[State](
    path: str | Path,
    template: State,
    runner: ChunkRunner[State] | Callable[[State], State],
    *,
    chunks: int,
    expected_metadata: Mapping[str, object] | None = None,
) -> list[State]:
    """Restore a checkpoint and re-run ``chunks`` runner calls from it.

    Every RNG stream (environment, drift, agent) lives in the saved carry, so
    the replayed segment repeats the original one exactly on the same software
    and backend; it is a debugging and audit tool, not a cross-device promise.
    ``runner`` is any :class:`perpetua.runtime.ChunkRunner` whose carry is
    complete (a lifetime or learner runner, bare or wrapped); its ``metadata``
    is the expected configuration unless ``expected_metadata`` is given. The
    exception is :class:`perpetua.torch_bridge.TorchRunner`, refused with
    ``TypeError``: its agent's state lives in the PyTorch agent, outside the
    carry, so a replay would continue from whatever the agent holds now.
    Restore it with ``TorchRunner.restore`` and call the runner. A plain
    ``carry -> carry`` callable is accepted with ``expected_metadata`` and is
    taken not to donate. Returns the carry after each chunk. A runner whose
    ``donates`` is True consumes its input, so each returned carry is copied
    before it is passed on.
    """
    chunks = as_int(chunks, "chunks must be a positive integer.", minimum=1)
    _require_complete_carry(runner, template)
    metadata = (
        expected_metadata if expected_metadata is not None else getattr(runner, "metadata", None)
    )
    if metadata is None:
        raise ValueError("Pass expected_metadata for a runner without metadata.")
    carry = restore(path, template, expected_metadata=cast(Mapping[str, object], metadata))
    segment: list[State] = []
    donates = bool(getattr(runner, "donates", False))
    for _ in range(chunks):
        carry = runner(carry)
        segment.append(carry)
        if donates:  # the next call consumes its input; keep the returned chunk intact
            carry = jax.tree.map(_copy_leaf, carry)
    return segment
