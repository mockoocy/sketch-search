import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt

SECRET = os.getenv("SECRET_KEY", "dev-insecure-change")
ALGO = "HS256"
ACCESS_MIN = int(os.getenv("ACCESS_TOKEN_MIN", "30"))


def make_token(sub: str, role: str = "user") -> str:
    exp = datetime.now(UTC) + timedelta(minutes=ACCESS_MIN)
    return jwt.encode({"sub": sub, "role": role, "exp": exp}, SECRET, algorithm=ALGO)


def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGO])
    except jwt.PyJWTError:
        return None


def gen_code(n: int) -> str:
    return f"{secrets.randbelow(10**n):0{n}d}"


def new_salt_hex() -> str:
    return secrets.token_hex(16)


def hash_code(code: str, salt_hex: str) -> str:
    msg = bytes.fromhex(salt_hex) + code.encode()
    return hmac.new(SECRET.encode(), msg, hashlib.sha256).hexdigest()
