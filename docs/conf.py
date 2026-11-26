"""Sphinx configuration for the Perpetua documentation site."""

from __future__ import annotations

project = "Perpetua"
author = "Perpetua contributors"
release = "0.0.1a0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".md": "markdown"}
master_doc = "index"
html_theme = "furo"
html_title = "Perpetua"
html_static_path = []

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
# Optional extras need not be installed to build the API reference.
autodoc_mock_imports = ["torch"]
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# JAX/Flax signatures are intentionally loose; do not fail on missing
# cross-references to third-party types, but do fail on any other warning.
nitpicky = False
suppress_warnings = ["ref.python"]


def index_viewcode_backlinks(app):
    """Point each ``[docs]`` link of a source page at the page that indexes the object.

    viewcode links back to the page it read last. The environment pages are
    read after the API reference and show their classes with ``:no-index:``,
    so they hold no anchor for such a link to land on.
    """
    indexed = app.env.get_domain("py").objects
    for entry in getattr(app.env, "_viewcode_modules", {}).values():
        if not entry:
            continue
        _, _, used, module = entry
        for name in used:
            if f"{module}.{name}" in indexed:
                used[name] = indexed[f"{module}.{name}"].docname
    return []


def setup(app):
    # Before viewcode writes the source pages (default priority 500).
    app.connect("html-collect-pages", index_viewcode_backlinks, priority=400)
