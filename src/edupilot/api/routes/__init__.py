"""
Routers
=======
One module per resource. `ROUTERS` is the registration order used by
`edupilot.api.app.create_app` — `system` is last so its catch-all `/` never
shadows an API path.
"""

from . import auth, chat, evaluation, knowledge_base, self_study, sessions, system

#: Registration order. Every router must appear here to be served.
ROUTERS = [
    auth.router,
    chat.router,
    sessions.router,
    knowledge_base.router,
    self_study.router,
    evaluation.router,
    system.router,
]

__all__ = ["ROUTERS"]
