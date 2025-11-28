"""
Authentication and authorization for the Congressional Trading Intelligence API.
Implements API key authentication with tiered rate limiting.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict
import time

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .config import get_settings, RATE_LIMITS
from .schemas import AccessTierEnum

settings = get_settings()

# API Key header definition
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


class APIKeyData(BaseModel):
    """Data associated with an API key."""
    id: int
    name: str
    key_hash: str
    tier: AccessTierEnum
    user_id: Optional[int] = None
    is_active: bool = True
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


class RateLimitInfo(BaseModel):
    """Rate limit tracking information."""
    requests_today: int = 0
    requests_this_hour: int = 0
    last_request: Optional[datetime] = None
    daily_reset: datetime
    hourly_reset: datetime


# In-memory storage for API keys and rate limits
# In production, use Redis or database
_api_keys_store: Dict[str, APIKeyData] = {}
_rate_limits_store: Dict[str, RateLimitInfo] = defaultdict(lambda: RateLimitInfo(
    daily_reset=datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1),
    hourly_reset=datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
))
_usage_store: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

# Pre-configured API keys for different tiers (for demo purposes)
# In production, these would be stored in a database
_default_keys = {
    "demo-public-key-12345": APIKeyData(
        id=1,
        name="Demo Public Key",
        key_hash=hashlib.sha256("demo-public-key-12345".encode()).hexdigest(),
        tier=AccessTierEnum.PUBLIC,
        is_active=True,
        created_at=datetime.utcnow(),
    ),
    "demo-journalist-key-67890": APIKeyData(
        id=2,
        name="Demo Journalist Key",
        key_hash=hashlib.sha256("demo-journalist-key-67890".encode()).hexdigest(),
        tier=AccessTierEnum.JOURNALIST,
        is_active=True,
        created_at=datetime.utcnow(),
    ),
    "demo-academic-key-11111": APIKeyData(
        id=3,
        name="Demo Academic Key",
        key_hash=hashlib.sha256("demo-academic-key-11111".encode()).hexdigest(),
        tier=AccessTierEnum.ACADEMIC,
        is_active=True,
        created_at=datetime.utcnow(),
    ),
    "demo-premium-key-99999": APIKeyData(
        id=4,
        name="Demo Premium Key",
        key_hash=hashlib.sha256("demo-premium-key-99999".encode()).hexdigest(),
        tier=AccessTierEnum.PREMIUM,
        is_active=True,
        created_at=datetime.utcnow(),
    ),
}

# Initialize with default keys
_api_keys_store.update(_default_keys)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new secure API key."""
    return secrets.token_urlsafe(32)


def create_api_key(
    name: str,
    tier: AccessTierEnum = AccessTierEnum.PUBLIC,
    user_id: Optional[int] = None,
    expires_days: int = 365
) -> Tuple[str, APIKeyData]:
    """
    Create a new API key.

    Args:
        name: Human-readable name for the key
        tier: Access tier (determines rate limits)
        user_id: Optional user ID to associate with the key
        expires_days: Number of days until key expires

    Returns:
        Tuple of (raw_key, key_data)
    """
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)

    key_data = APIKeyData(
        id=len(_api_keys_store) + 1,
        name=name,
        key_hash=key_hash,
        tier=tier,
        user_id=user_id,
        is_active=True,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None,
    )

    _api_keys_store[raw_key] = key_data
    return raw_key, key_data


def validate_api_key(api_key: str) -> Optional[APIKeyData]:
    """
    Validate an API key and return its data.

    Args:
        api_key: The raw API key to validate

    Returns:
        APIKeyData if valid, None otherwise
    """
    if not api_key:
        return None

    # Check if key exists
    key_data = _api_keys_store.get(api_key)
    if not key_data:
        return None

    # Check if key is active
    if not key_data.is_active:
        return None

    # Check if key has expired
    if key_data.expires_at and key_data.expires_at < datetime.utcnow():
        return None

    # Update last used timestamp
    key_data.last_used = datetime.utcnow()

    return key_data


def check_rate_limit(api_key: str, tier: AccessTierEnum) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if request is within rate limits.

    Args:
        api_key: The API key making the request
        tier: The access tier of the key

    Returns:
        Tuple of (allowed, rate_limit_info)
    """
    now = datetime.utcnow()
    rate_info = _rate_limits_store[api_key]

    # Reset daily counter if needed
    if now >= rate_info.daily_reset:
        rate_info.requests_today = 0
        rate_info.daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    # Reset hourly counter if needed
    if now >= rate_info.hourly_reset:
        rate_info.requests_this_hour = 0
        rate_info.hourly_reset = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Get rate limit for tier
    daily_limit = RATE_LIMITS.get(tier.value, RATE_LIMITS["public"])

    # Check if within limits
    allowed = rate_info.requests_today < daily_limit

    # Prepare response headers info
    headers_info = {
        "X-RateLimit-Limit": str(daily_limit),
        "X-RateLimit-Remaining": str(max(0, daily_limit - rate_info.requests_today)),
        "X-RateLimit-Reset": str(int(rate_info.daily_reset.timestamp())),
    }

    if allowed:
        # Increment counters
        rate_info.requests_today += 1
        rate_info.requests_this_hour += 1
        rate_info.last_request = now

    return allowed, headers_info


def record_endpoint_usage(api_key: str, endpoint: str) -> None:
    """Record usage of a specific endpoint."""
    _usage_store[api_key][endpoint] += 1


def get_usage_stats(api_key: str) -> Dict[str, Any]:
    """Get usage statistics for an API key."""
    rate_info = _rate_limits_store.get(api_key)
    key_data = _api_keys_store.get(api_key)

    if not key_data:
        return {}

    daily_limit = RATE_LIMITS.get(key_data.tier.value, RATE_LIMITS["public"])

    return {
        "api_key_id": key_data.id,
        "tier": key_data.tier.value,
        "rate_limit": daily_limit,
        "requests_today": rate_info.requests_today if rate_info else 0,
        "requests_this_hour": rate_info.requests_this_hour if rate_info else 0,
        "endpoints_used": dict(_usage_store.get(api_key, {})),
    }


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    key_data = _api_keys_store.get(api_key)
    if key_data:
        key_data.is_active = False
        return True
    return False


def rotate_api_key(old_key: str) -> Optional[Tuple[str, APIKeyData]]:
    """
    Rotate an API key (create new, revoke old).

    Args:
        old_key: The current API key to rotate

    Returns:
        Tuple of (new_raw_key, new_key_data) or None if old key not found
    """
    old_data = _api_keys_store.get(old_key)
    if not old_data:
        return None

    # Create new key with same properties
    new_key, new_data = create_api_key(
        name=f"{old_data.name} (rotated)",
        tier=old_data.tier,
        user_id=old_data.user_id,
    )

    # Revoke old key
    revoke_api_key(old_key)

    return new_key, new_data


async def get_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[APIKeyData]:
    """
    FastAPI dependency to extract and validate API key.
    Returns None for unauthenticated requests (allows public endpoints).

    Usage:
        @app.get("/endpoint")
        async def endpoint(api_key: Optional[APIKeyData] = Depends(get_api_key)):
            if api_key:
                # Authenticated request
            else:
                # Unauthenticated request
    """
    if not api_key:
        return None

    key_data = validate_api_key(api_key)
    if key_data:
        # Record endpoint usage
        record_endpoint_usage(api_key, request.url.path)

    return key_data


async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> APIKeyData:
    """
    FastAPI dependency that requires a valid API key.
    Raises HTTPException if key is missing or invalid.

    Usage:
        @app.get("/protected")
        async def protected(api_key: APIKeyData = Depends(require_api_key)):
            # Only authenticated requests reach here
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Pass via X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_data = validate_api_key(api_key)
    if not key_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check rate limit
    allowed, rate_info = check_rate_limit(api_key, key_data.tier)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Daily limit: {rate_info['X-RateLimit-Limit']} requests",
            headers=rate_info,
        )

    # Record endpoint usage
    record_endpoint_usage(api_key, request.url.path)

    return key_data


async def require_tier(
    required_tiers: list[AccessTierEnum]
):
    """
    Factory for creating tier-specific authorization dependencies.

    Usage:
        @app.post("/stories/generate")
        async def generate_story(
            api_key: APIKeyData = Depends(require_tier([AccessTierEnum.JOURNALIST, AccessTierEnum.PREMIUM]))
        ):
            # Only journalist or premium tier can access
    """
    async def check_tier(api_key: APIKeyData = Depends(require_api_key)) -> APIKeyData:
        if api_key.tier not in required_tiers:
            tier_names = ", ".join(t.value for t in required_tiers)
            raise HTTPException(
                status_code=403,
                detail=f"This endpoint requires {tier_names} tier access",
            )
        return api_key

    return check_tier


# Convenience dependencies for specific tiers
require_journalist = require_tier([AccessTierEnum.JOURNALIST, AccessTierEnum.ACADEMIC, AccessTierEnum.PREMIUM])
require_academic = require_tier([AccessTierEnum.ACADEMIC, AccessTierEnum.PREMIUM])
require_premium = require_tier([AccessTierEnum.PREMIUM])
