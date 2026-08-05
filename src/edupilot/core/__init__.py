"""
Core
====
Cross-cutting concerns that every other package depends on:

  * `config`        — settings resolved from the environment
  * `services`      — the composition root holding every singleton
  * `observability` — structured logging and request-id propagation

`config` is safe to import from anywhere. `services` is not imported here,
because touching it builds nothing but does pull in the retrieval stack;
import it directly (`from edupilot.core.services import services`) at the
point of use.
"""

from .observability import RequestContext, configure_logging, request_id_var, timed

__all__ = ["RequestContext", "configure_logging", "request_id_var", "timed"]
