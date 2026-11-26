"""Collection rules shared by the test modules."""

import importlib.util

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give every torch-gated test the ``torch`` marker and skip it without the extra.

    A test is torch-gated when it carries ``pytest.mark.torch`` or its module
    imports torch at the top. ``pytest.mark.torch(partial=True)`` is for a
    test that runs without torch but then leaves a torch part out: it is
    selected by ``-m torch`` and never skipped here.
    """
    installed = importlib.util.find_spec("torch") is not None
    for item in items:
        module = getattr(item, "module", None)
        if getattr(getattr(module, "torch", None), "__name__", None) == "torch":
            item.add_marker(pytest.mark.torch)
        marker = item.get_closest_marker("torch")
        if marker is not None and not installed and not marker.kwargs.get("partial", False):
            item.add_marker(pytest.mark.skip(reason="optional torch extra"))
