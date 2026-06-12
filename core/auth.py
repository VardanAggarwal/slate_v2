"""User authentication against the users table (AUTH.md §2). bcrypt hashes.

Used by both credential checks in the system: the MCP OAuth login page
(mcp_server.py) and the Basic-Auth /status page (server.py).

The env AUTH_USER/AUTH_PASS pair survives only as a bootstrap: while the
users table is empty, a login matching the env pair auto-provisions that
user as the first admin. Once any user row exists, the env path is dead.
"""
import secrets

import bcrypt

from core import config, store


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def check_password(password: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("ascii"))
    except ValueError:  # malformed hash must fail closed, not 500
        return False


def authenticate(conn, username: str, password: str):
    """Validate credentials; return the users row on success, else None."""
    row = store.get_user_by_username(conn, username)
    if row:
        return row if check_password(password, row["pw_hash"]) else None
    if (store.count_users(conn) == 0
            and config.AUTH_USER and config.AUTH_PASS
            and secrets.compare_digest(username, config.AUTH_USER)
            and secrets.compare_digest(password, config.AUTH_PASS)):
        with conn:
            store.create_user(conn, username, hash_password(password), is_admin=True)
        return store.get_user_by_username(conn, username)
    return None
