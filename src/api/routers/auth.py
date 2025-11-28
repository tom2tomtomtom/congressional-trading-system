"""
Auth API router - Authentication and API key management.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..auth import (
    get_api_key,
    require_api_key,
    create_api_key,
    revoke_api_key,
    rotate_api_key,
    get_usage_stats,
    APIKeyData,
    _api_keys_store,
)
from ..schemas import (
    APIKeyCreate,
    APIKeyResponse,
    APIKeyInfo,
    UsageStats,
    AccessTierEnum,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/keys",
    response_model=APIKeyResponse,
    summary="Create API key",
    description="""
    Create a new API key for accessing the API.

    **Access Tiers:**
    - `public`: 100 requests/day, basic read access
    - `journalist`: 1,000 requests/day, full access + story generation
    - `academic`: 5,000 requests/day, bulk export + research features
    - `premium`: Unlimited, all features + priority support

    The API key is only shown once on creation. Store it securely.
    """,
    responses={
        201: {"description": "API key created successfully"},
        400: {"description": "Invalid request"},
    },
)
async def create_key(
    request: APIKeyCreate,
) -> APIKeyResponse:
    """Create a new API key."""
    raw_key, key_data = create_api_key(
        name=request.name,
        tier=request.tier,
        expires_days=request.expires_days or 365,
    )

    return APIKeyResponse(
        id=key_data.id,
        name=key_data.name,
        key=raw_key,  # Only shown once!
        tier=key_data.tier,
        rate_limit=_get_rate_limit(key_data.tier),
        expires_at=key_data.expires_at,
        created_at=key_data.created_at,
    )


@router.get(
    "/keys/me",
    response_model=APIKeyInfo,
    summary="Get current API key info",
    description="""
    Get information about the current API key.

    Includes usage statistics and rate limit information.
    """,
)
async def get_current_key_info(
    request: Request,
    api_key: APIKeyData = Depends(require_api_key),
) -> APIKeyInfo:
    """Get information about the current API key."""
    # Find the raw key from the request
    raw_key = request.headers.get("X-API-Key", "")
    usage = get_usage_stats(raw_key)

    return APIKeyInfo(
        id=api_key.id,
        name=api_key.name,
        key_prefix=raw_key[:8] + "..." if raw_key else "...",
        tier=api_key.tier,
        rate_limit=_get_rate_limit(api_key.tier),
        requests_today=usage.get("requests_today", 0),
        requests_this_month=usage.get("requests_today", 0) * 30,  # Approximate
        last_used=api_key.last_used,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        is_active=api_key.is_active,
    )


@router.get(
    "/usage",
    response_model=UsageStats,
    summary="Get usage statistics",
    description="""
    Get detailed usage statistics for the current API key.

    Includes:
    - Request counts (today, this week, this month)
    - Rate limit information
    - Endpoints used
    """,
)
async def get_usage(
    request: Request,
    api_key: APIKeyData = Depends(require_api_key),
) -> UsageStats:
    """Get usage statistics for the current API key."""
    raw_key = request.headers.get("X-API-Key", "")
    usage = get_usage_stats(raw_key)

    return UsageStats(
        api_key_id=api_key.id,
        requests_today=usage.get("requests_today", 0),
        requests_this_week=usage.get("requests_today", 0) * 7,  # Approximate
        requests_this_month=usage.get("requests_today", 0) * 30,  # Approximate
        rate_limit=_get_rate_limit(api_key.tier),
        tier=api_key.tier,
        endpoints_used=usage.get("endpoints_used", {}),
    )


@router.post(
    "/keys/rotate",
    response_model=APIKeyResponse,
    summary="Rotate API key",
    description="""
    Rotate the current API key (create new, revoke old).

    The old key is immediately revoked and a new key is generated
    with the same permissions. Store the new key securely.
    """,
)
async def rotate_key(
    request: Request,
    api_key: APIKeyData = Depends(require_api_key),
) -> APIKeyResponse:
    """Rotate the current API key."""
    raw_key = request.headers.get("X-API-Key", "")

    result = rotate_api_key(raw_key)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to rotate key")

    new_key, new_data = result

    return APIKeyResponse(
        id=new_data.id,
        name=new_data.name,
        key=new_key,  # Only shown once!
        tier=new_data.tier,
        rate_limit=_get_rate_limit(new_data.tier),
        expires_at=new_data.expires_at,
        created_at=new_data.created_at,
    )


@router.delete(
    "/keys/me",
    summary="Revoke current API key",
    description="""
    Revoke the current API key.

    Once revoked, the key cannot be used for any further requests.
    This action cannot be undone.
    """,
    responses={
        200: {"description": "API key revoked successfully"},
        404: {"description": "API key not found"},
    },
)
async def revoke_current_key(
    request: Request,
    api_key: APIKeyData = Depends(require_api_key),
) -> dict:
    """Revoke the current API key."""
    raw_key = request.headers.get("X-API-Key", "")

    success = revoke_api_key(raw_key)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"message": "API key revoked successfully", "key_id": api_key.id}


@router.get(
    "/tiers",
    summary="Get available access tiers",
    description="""
    Get information about available API access tiers.

    Each tier has different rate limits and feature access.
    """,
)
async def get_tiers() -> dict:
    """Get available access tiers."""
    from ..config import RATE_LIMITS

    return {
        "tiers": [
            {
                "name": "public",
                "rate_limit_daily": RATE_LIMITS["public"],
                "features": [
                    "Read members",
                    "Read trades",
                    "View leaderboards",
                    "View statistics",
                ],
            },
            {
                "name": "journalist",
                "rate_limit_daily": RATE_LIMITS["journalist"],
                "features": [
                    "All public features",
                    "Conviction score analysis",
                    "Timing analysis",
                    "Story generation",
                    "Member swamp scores",
                ],
            },
            {
                "name": "academic",
                "rate_limit_daily": RATE_LIMITS["academic"],
                "features": [
                    "All journalist features",
                    "Bulk data export",
                    "Historical analysis",
                    "Research endpoints",
                ],
            },
            {
                "name": "premium",
                "rate_limit_daily": RATE_LIMITS["premium"],
                "features": [
                    "All academic features",
                    "Unlimited requests",
                    "Priority support",
                    "Custom integrations",
                    "Webhook subscriptions",
                ],
            },
        ]
    }


def _get_rate_limit(tier: AccessTierEnum) -> int:
    """Get rate limit for a tier."""
    from ..config import RATE_LIMITS
    return RATE_LIMITS.get(tier.value, RATE_LIMITS["public"])
