"""
HTTP layer
==========
FastAPI application, request/response schemas, shared dependencies, and one
router module per resource.

`app` is intentionally not re-exported here: importing it builds the whole
application (and touches the database on lifespan), which is too much of a
side effect for `import edupilot.api`. Reach for it explicitly::

    from edupilot.api.app import app, create_app
"""

__all__: list[str] = []
