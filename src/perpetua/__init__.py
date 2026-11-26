"""Perpetua: a lifetime-first interface for continual reinforcement learning."""

from perpetua import (
    checkpoint,
    compute,
    drift,
    evaluation,
    measurement,
    metrics,
    params,
    policies,
    presets,
    probes,
    protocol,
    reporting,
    runtime,
    spaces,
    tasks,
    testing,
    tuning,
)
from perpetua.compose import compose
from perpetua.core import Clock, Environment, Info, Metadata, OneHotObservation, Oracle
from perpetua.registration import (
    EnvSpec,
    environments,
    make,
    register,
    registered,
    spec,
    unregister,
)


def __getattr__(name: str) -> object:
    """Load ``pp.agents`` on first use; it needs the optional ``training`` extra."""
    if name == "agents":
        import importlib

        return importlib.import_module("perpetua.agents")
    raise AttributeError(f"module 'perpetua' has no attribute {name!r}")


__all__ = [
    "Clock",
    "EnvSpec",
    "Environment",
    "Info",
    "Metadata",
    "OneHotObservation",
    "Oracle",
    "checkpoint",
    "compose",
    "compute",
    "drift",
    "environments",
    "evaluation",
    "make",
    "measurement",
    "metrics",
    "params",
    "policies",
    "presets",
    "register",
    "registered",
    "probes",
    "protocol",
    "reporting",
    "runtime",
    "spaces",
    "spec",
    "tasks",
    "testing",
    "tuning",
    "unregister",
]
