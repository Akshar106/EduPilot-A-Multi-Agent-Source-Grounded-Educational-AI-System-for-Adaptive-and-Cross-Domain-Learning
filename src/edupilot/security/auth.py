"""
Authentication and authorization
================================
JWT auth with per-user data scoping and a role for knowledge-base admins.

The previous API had no authentication of any kind, and `session_id` was
supplied by the client and trusted. Combined with `GET /api/sessions`, which
returned the twenty most recent session IDs to anyone, that made every
student's chat history and uploaded documents readable — and deletable — by
any visitor:

    GET  /api/sessions                     -> harvest IDs
    GET  /api/sessions/<id>                -> read that person's transcript
    DELETE /api/sessions/<id>              -> destroy it

Identity is the only real fix. A session now belongs to a user id taken from
a signed token, never from the request body, and every read and write is
scoped to the caller.

Design notes:

  * Access tokens are short-lived and stateless; refresh tokens are long-lived
    and stored server-side so logout can actually revoke them. A stateless
    refresh token cannot be revoked, which makes "sign out everywhere"
    impossible.
  * Passwords use bcrypt with a per-password salt.
  * `verify_password` runs a dummy hash when the user does not exist, so
    response timing does not reveal which accounts are registered.
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import bcrypt
import jwt

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(minutes=30)
REFRESH_TOKEN_TTL = timedelta(days=14)

#: bcrypt truncates at 72 bytes; reject longer rather than silently ignoring
#: the tail, which would make two different long passwords equivalent.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 10

#: Cost factor. 12 is ~250ms on current hardware — slow enough to matter for
#: offline cracking, fast enough for interactive login.
BCRYPT_ROUNDS = 12

#: Precomputed hash used to equalize timing on unknown-user login attempts.
_DUMMY_HASH = bcrypt.hashpw(b"timing-equalization-placeholder", bcrypt.gensalt(BCRYPT_ROUNDS))


class Role(str, Enum):
    """
    Authorization roles.

    `str, Enum` rather than `StrEnum` so this runs on Python 3.10, where
    `enum.StrEnum` does not exist. The mixin gives the same behaviour that
    matters here: a Role serializes as its value in JSON and compares equal
    to the plain string.
    """

    STUDENT = "student"
    """Can chat and manage their own study sessions and uploads."""
    ADMIN = "admin"
    """Additionally can modify the shared course knowledge base."""

    def __str__(self) -> str:
        return self.value


class AuthError(Exception):
    """Authentication or authorization failure."""

    def __init__(self, message: str, *, status: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class User:
    """An authenticated principal."""

    user_id: str
    email: str
    role: Role
    display_name: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def namespace(self) -> str:
        """Per-user vector namespace prefix, keeping study uploads isolated."""
        return f"u_{self.user_id.replace('-', '')}"


# ---------------------------------------------------------------------------
# Secret
# ---------------------------------------------------------------------------


def get_secret_key() -> str:
    """
    Load the JWT signing secret.

    Refuses to start without one in production. An auto-generated fallback in
    development is a convenience; the same behaviour in production would mean
    every restart silently invalidated all tokens, and a hardcoded default
    would mean anyone could mint an admin token.
    """
    secret = os.getenv("JWT_SECRET_KEY", "")
    if secret:
        if len(secret) < 32:
            raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters")
        return secret

    if os.getenv("EDUPILOT_ENV", "development").lower() == "production":
        raise RuntimeError(
            "JWT_SECRET_KEY is required in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    logger.warning(
        "JWT_SECRET_KEY not set — using an ephemeral development key. "
        "All tokens become invalid when this process restarts."
    )
    return secrets.token_urlsafe(48)


_SECRET: str | None = None
_secret_lock = threading.Lock()


def _secret() -> str:
    global _SECRET
    if _SECRET is None:
        with _secret_lock:
            if _SECRET is None:
                _SECRET = get_secret_key()
    return _SECRET


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a password with bcrypt. Raises ValueError if it fails policy."""
    encoded = password.encode("utf-8")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes "
            "(bcrypt silently truncates beyond this)."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt(BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    """
    Check a password against its hash in constant-ish time.

    When `password_hash` is None (no such user) a dummy comparison still runs,
    so a missing account and a wrong password take the same time. Skipping it
    turns login into a user-enumeration oracle.
    """
    encoded = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    if password_hash is None:
        bcrypt.checkpw(encoded, _DUMMY_HASH)
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        logger.warning("malformed password hash encountered")
        return False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def create_access_token(user: User, *, ttl: timedelta = ACCESS_TOKEN_TTL) -> str:
    """Mint a short-lived stateless access token."""
    now = datetime.now(UTC)
    payload = {
        "sub": user.user_id,
        "email": user.email,
        "role": str(user.role),
        "name": user.display_name,
        "type": "access",
        "iat": now,
        "exp": now + ttl,
        "jti": secrets.token_urlsafe(8),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def create_refresh_token() -> tuple[str, str, datetime]:
    """
    Mint a refresh token.

    Returns (token, token_hash, expires_at). Only the hash is stored, so a
    database leak does not hand out usable refresh tokens.
    """
    import hashlib

    token = secrets.token_urlsafe(48)
    digest = hashlib.sha256(token.encode()).hexdigest()
    return token, digest, datetime.now(UTC) + REFRESH_TOKEN_TTL


def decode_access_token(token: str) -> User:
    """
    Verify an access token and return its principal.

    Raises:
        AuthError: expired, malformed, wrong type, or bad signature.
    """
    try:
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired. Please sign in again.") from None
    except jwt.InvalidTokenError as exc:
        logger.info("rejected token: %s", exc)
        raise AuthError("Invalid authentication token.") from None

    # An access endpoint must not accept a refresh token.
    if payload.get("type") != "access":
        raise AuthError("Invalid token type.")

    try:
        role = Role(payload.get("role", Role.STUDENT))
    except ValueError:
        role = Role.STUDENT

    return User(
        user_id=str(payload["sub"]),
        email=str(payload.get("email", "")),
        role=role,
        display_name=str(payload.get("name", "")),
    )


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------


class UserStore:
    """SQLite-backed user and refresh-token storage."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        user_id       TEXT PRIMARY KEY,
        email         TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'student',
        display_name  TEXT NOT NULL DEFAULT '',
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        last_login_at TEXT,
        is_active     INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS refresh_tokens (
        token_hash TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        revoked    INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # Several components hold connections to this file; without a busy
            # timeout, routine contention raises "database is locked" at once.
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """
        Transaction that commits on success and **rolls back on failure**.

        The rollback is the point. Python's sqlite3 opens a transaction
        implicitly on INSERT/UPDATE/DELETE; if the statement raises and
        nothing rolls back, that connection keeps an open write transaction
        for its whole lifetime and every other connection to the file then
        fails with "database is locked".

        A duplicate-email registration hits exactly that path, so one
        rejected signup would wedge the entire application until restart.
        """
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -- users ------------------------------------------------------------

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    def create_user(
        self, email: str, password: str, *, role: Role = Role.STUDENT, display_name: str = ""
    ) -> User:
        """
        Register a user.

        Raises:
            AuthError: the email is already registered.
            ValueError: the password fails policy.
        """
        email = self.normalize_email(email)
        if not email or "@" not in email:
            raise ValueError("A valid email address is required.")

        password_hash = hash_password(password)
        user_id = str(uuid.uuid4())
        try:
            with self._write() as conn:
                conn.execute(
                    "INSERT INTO users (user_id, email, password_hash, role, display_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, email, password_hash, str(role),
                     display_name or email.split("@")[0]),
                )
        except sqlite3.IntegrityError:
            raise AuthError("An account with that email already exists.", status=409) from None

        logger.info("registered user %s (%s)", user_id, role)
        return User(user_id=user_id, email=email, role=role, display_name=display_name)

    def authenticate(self, email: str, password: str) -> User:
        """
        Verify credentials.

        Raises:
            AuthError: with a message that does not reveal whether the account
                exists — distinguishing "no such user" from "wrong password"
                hands an attacker a list of valid accounts.
        """
        row = self._conn().execute(
            "SELECT * FROM users WHERE email=?", (self.normalize_email(email),)
        ).fetchone()

        stored_hash = row["password_hash"] if row else None
        if not verify_password(password, stored_hash) or not row:
            raise AuthError("Incorrect email or password.")
        if not row["is_active"]:
            raise AuthError("This account has been deactivated.", status=403)

        with self._write() as conn:
            conn.execute(
                "UPDATE users SET last_login_at=datetime('now') WHERE user_id=?",
                (row["user_id"],),
            )

        return User(
            user_id=row["user_id"],
            email=row["email"],
            role=Role(row["role"]),
            display_name=row["display_name"],
        )

    def get_user(self, user_id: str) -> User | None:
        row = self._conn().execute(
            "SELECT * FROM users WHERE user_id=? AND is_active=1", (user_id,)
        ).fetchone()
        if not row:
            return None
        return User(
            user_id=row["user_id"],
            email=row["email"],
            role=Role(row["role"]),
            display_name=row["display_name"],
        )

    def count_users(self) -> int:
        return self._conn().execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]

    # -- refresh tokens ---------------------------------------------------

    def store_refresh_token(self, user_id: str, token_hash: str, expires_at: datetime) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token_hash, user_id, expires_at) "
                "VALUES (?, ?, ?)",
                (token_hash, user_id, expires_at.isoformat()),
            )

    def consume_refresh_token(self, token: str) -> User:
        """
        Exchange a refresh token for its user, rotating it out.

        Single-use: the token is revoked on redemption, so a stolen token is
        usable at most once and reuse is detectable.
        """
        import hashlib

        digest = hashlib.sha256(token.encode()).hexdigest()
        row = self._conn().execute(
            "SELECT * FROM refresh_tokens WHERE token_hash=?", (digest,)
        ).fetchone()

        if not row or row["revoked"]:
            raise AuthError("Invalid or already-used refresh token.")
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(UTC):
            raise AuthError("Refresh token expired. Please sign in again.")

        with self._write() as conn:
            conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE token_hash=?", (digest,))

        user = self.get_user(row["user_id"])
        if not user:
            raise AuthError("Account no longer exists.")
        return user

    def revoke_all_refresh_tokens(self, user_id: str) -> int:
        """Sign out everywhere. Returns the number of tokens revoked."""
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE refresh_tokens SET revoked=1 WHERE user_id=? AND revoked=0", (user_id,)
            )
            return cur.rowcount or 0

    def purge_expired_tokens(self) -> int:
        with self._write() as conn:
            cur = conn.execute(
                "DELETE FROM refresh_tokens WHERE expires_at < ?",
                (datetime.now(UTC).isoformat(),),
            )
            return cur.rowcount or 0
