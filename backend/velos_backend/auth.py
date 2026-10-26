"""API-key authentication (M2 scope: 'basic authentication/authorization as agreed').

Clients send ``X-API-Key: <key>``. Keys and roles are configured through the
``VELOS_API_KEYS`` environment variable, never in code. Roles:

* ``user``  – can upload datasets, submit jobs and read its own resources
* ``admin`` – can also read every user's resources and cancel any job
"""
from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from .config import ApiKeyInfo, get_settings

log = logging.getLogger("velos.auth")
_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_header)) -> ApiKeyInfo:
    settings = get_settings()
    if settings.insecure_default_key:
        log.warning("VELOS_API_KEYS is not set — using the insecure development key")
    if not key or key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="missing or invalid API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return settings.api_keys[key]


def require_admin(info: ApiKeyInfo = Depends(require_api_key)) -> ApiKeyInfo:
    if info.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin role required")
    return info


def can_access(info: ApiKeyInfo, owner: str) -> bool:
    return info.role == "admin" or info.name == owner
