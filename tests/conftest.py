"""
Test configuration
==================
Redirects EDUPILOT_DATA_DIR at a throwaway directory *before* any edupilot
module is imported, so a test run can never read or write the developer's
real database, corpus, or caches.

This has to happen at import time of conftest — `edupilot.core.config`
resolves its paths once, at module import, and every other module reads them
from there.
"""

from __future__ import annotations

import os
import tempfile

_TMP_DATA = tempfile.mkdtemp(prefix="edupilot-test-data-")
os.environ["EDUPILOT_DATA_DIR"] = _TMP_DATA
os.environ.setdefault("JWT_SECRET_KEY", "test-only-key-not-used-in-production-0123456789")
os.environ.setdefault("EDUPILOT_ENV", "development")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def data_dir() -> str:
    """The temporary DATA_DIR this run is pinned to."""
    return _TMP_DATA
