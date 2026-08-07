"""
Import integrity.

Every module must import, and every name a package promises in `__all__` must
actually resolve. Package `__init__` files re-export names that are unused in
the module that defines them, so an automated "remove unused imports" pass can
delete a re-export and break `from edupilot.agents import X` at runtime while
every other test still passes. These tests are the tripwire for that.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import edupilot

#: Packages whose __init__ re-exports a public surface.
PACKAGES_WITH_REEXPORTS = [
    "edupilot.agents",
    "edupilot.db",
    "edupilot.evaluation",
    "edupilot.guardrails",
    "edupilot.ingestion",
    "edupilot.llm",
    "edupilot.retrieval",
    "edupilot.security",
]


def _all_modules() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(edupilot.__path__, "edupilot."))


def test_package_discovery_is_not_empty():
    """Guards against the walk silently finding nothing and vacuously passing."""
    modules = _all_modules()
    assert len(modules) > 30, f"only found {len(modules)} modules — discovery is broken"


@pytest.mark.parametrize("module_name", _all_modules())
def test_module_imports(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("package_name", PACKAGES_WITH_REEXPORTS)
def test_declared_exports_resolve(package_name):
    """
    Every name in __all__ must exist on the package.

    This is what catches a deleted re-export: `edupilot.agents.__init__` lists
    `is_refusal`, which it imports from `.contracts`, which in turn re-exports
    it from `edupilot.guardrails.refusal`. Remove that import as "unused" and
    importing edupilot.agents raises ImportError.
    """
    pkg = importlib.import_module(package_name)
    declared = getattr(pkg, "__all__", None)
    assert declared, f"{package_name} declares no __all__"

    missing = [name for name in declared if not hasattr(pkg, name)]
    assert not missing, f"{package_name}.__all__ promises missing names: {missing}"


def test_agents_exposes_the_refusal_protocol():
    """Explicit regression test for the exact break this file exists for."""
    from edupilot.agents import is_refusal, strip_refusal_marker

    assert callable(is_refusal)
    assert callable(strip_refusal_marker)


def test_importing_edupilot_is_cheap():
    """
    The top-level package must not drag in torch or open a network client.

    Services builds everything lazily; if that regresses, `import edupilot`
    starts loading models and the CLI's --help takes 30 seconds.
    """
    import os
    import pathlib
    import subprocess
    import sys

    # The subprocess does not inherit pytest's `pythonpath` setting, and an
    # editable-install .pth is not always honoured (see the iCloud note in
    # README troubleshooting). Point it at the same source tree this process
    # imported from, so the test measures import cost rather than path setup.
    src_dir = str(pathlib.Path(edupilot.__file__).resolve().parent.parent)

    code = (
        "import sys, edupilot;"
        "heavy=[m for m in ('torch','sentence_transformers','pinecone') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": src_dir},
    )
    assert out.returncode == 0, out.stderr
    assert not out.stdout.strip(), f"import edupilot eagerly loaded: {out.stdout.strip()}"
