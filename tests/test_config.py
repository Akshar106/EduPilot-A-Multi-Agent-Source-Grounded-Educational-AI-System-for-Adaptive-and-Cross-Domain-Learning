"""Configuration and path resolution."""

from __future__ import annotations

from pathlib import Path

from edupilot.core import config


def test_data_paths_live_under_data_dir():
    """Everything mutable resolves under DATA_DIR, so one mount covers it all."""
    data_dir = config.DATA_DIR
    for path in (
        config.KNOWLEDGE_BASE_DIR,
        config.SELF_STUDY_DIR,
        config.STATE_DIR,
        Path(config.SQLITE_DB_PATH),
        Path(config.EMBEDDING_CACHE_PATH),
        Path(config.SPARSE_ENCODER_PATH),
        Path(config.INDEX_POINTER_PATH),
    ):
        assert data_dir in path.parents or path == data_dir, f"{path} escapes DATA_DIR"


def test_data_dir_is_redirected_by_conftest():
    """The env override must win, or a test run would touch real data."""
    assert "edupilot-test-data-" in str(config.DATA_DIR)


def test_static_dir_lives_inside_the_package():
    """Web assets ship with the package, not with the working directory."""
    assert config.PACKAGE_DIR in config.STATIC_DIR.parents
    assert config.STATIC_DIR.name == "static"


def test_every_domain_is_fully_specified():
    required = {
        "name", "abbr", "color", "knowledge_base_path",
        "pinecone_namespace", "description", "keywords",
    }
    assert config.DOMAINS, "at least one domain must be configured"
    for key, domain in config.DOMAINS.items():
        missing = required - set(domain)
        assert not missing, f"domain {key} is missing {missing}"
        assert domain["keywords"], f"domain {key} has no keywords"


def test_domain_namespaces_are_unique():
    """Two domains sharing a namespace would silently merge their corpora."""
    namespaces = [d["pinecone_namespace"] for d in config.DOMAINS.values()]
    assert len(namespaces) == len(set(namespaces))


def test_knowledge_base_paths_are_unique_and_under_the_kb_dir():
    paths = [Path(d["knowledge_base_path"]) for d in config.DOMAINS.values()]
    assert len(paths) == len(set(paths))
    for path in paths:
        assert config.KNOWLEDGE_BASE_DIR in path.parents


def test_default_model_is_available():
    assert config.DEFAULT_MODEL in config.AVAILABLE_MODELS
    assert config.VERIFY_MODEL in config.AVAILABLE_MODELS


def test_groq_models_are_a_subset_of_available():
    assert set(config.GROQ_MODELS) <= set(config.AVAILABLE_MODELS)


def test_cors_is_not_a_wildcard():
    """allow_origins=['*'] with credentials lets any site call the API as the user."""
    assert "*" not in config.CORS_ALLOWED_ORIGINS
