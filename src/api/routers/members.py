"""
Members API router - Congressional member information and profiles.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import get_api_key, require_api_key, APIKeyData
from ..schemas import (
    MemberListResponse,
    MemberDetail,
    MemberSummary,
    TradeListResponse,
    TradeSummary,
    SwampScore,
    ChamberEnum,
    PartyEnum,
)
from ..services import MemberService, TradeService, AnalysisService

router = APIRouter(prefix="/members", tags=["members"])


@router.get(
    "",
    response_model=MemberListResponse,
    summary="List all congressional members",
    description="""
    Get a paginated list of all congressional members.

    Supports filtering by party, chamber, state, and search term.
    Results are sorted by last name by default.
    """,
)
async def list_members(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    party: Optional[str] = Query(None, description="Filter by party (D, R, I)"),
    chamber: Optional[str] = Query(None, description="Filter by chamber (house, senate)"),
    state: Optional[str] = Query(None, description="Filter by state (2-letter code)"),
    search: Optional[str] = Query(None, description="Search by name"),
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> MemberListResponse:
    """List all congressional members with optional filtering."""
    members, total = MemberService.get_members(
        page=page,
        per_page=per_page,
        party=party,
        chamber=chamber,
        state=state,
        search=search,
    )

    pages = (total + per_page - 1) // per_page

    return MemberListResponse(
        items=members,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1,
    )


@router.get(
    "/{member_id}",
    response_model=MemberDetail,
    summary="Get member details",
    description="""
    Get detailed information about a specific congressional member.

    Includes committee assignments, trading statistics, and contact information.
    """,
)
async def get_member(
    member_id: int,
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> MemberDetail:
    """Get detailed information about a member."""
    member = MemberService.get_member(member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    return member


@router.get(
    "/{member_id}/trades",
    response_model=TradeListResponse,
    summary="Get member's trades",
    description="""
    Get all stock trades disclosed by a specific member.

    Trades are sorted by transaction date in descending order.
    """,
)
async def get_member_trades(
    member_id: int,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> TradeListResponse:
    """Get trades for a specific member."""
    # Verify member exists
    member = MemberService.get_member(member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    trades, total = TradeService.get_member_trades(
        member_id=member_id,
        page=page,
        per_page=per_page,
    )

    pages = (total + per_page - 1) // per_page

    return TradeListResponse(
        items=trades,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1,
    )


@router.get(
    "/{member_id}/score",
    response_model=SwampScore,
    summary="Get member's swamp score",
    description="""
    Get the composite ethics/transparency score for a member.

    The Swamp Score (0-100) aggregates multiple factors:
    - Average conviction score on trades (40%)
    - Filing compliance rate (20%)
    - Oversight sector trading percentage (20%)
    - Timing suspicion patterns (10%)
    - Volume anomalies (10%)

    Higher scores indicate more concerning patterns.
    """,
    responses={
        200: {"description": "Swamp score retrieved successfully"},
        404: {"description": "Member not found"},
    },
)
async def get_member_score(
    member_id: int,
    api_key: APIKeyData = Depends(require_api_key),
) -> SwampScore:
    """Get swamp score for a specific member."""
    # Verify member exists
    member = MemberService.get_member(member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    score = AnalysisService.get_member_score(member_id)
    if not score:
        raise HTTPException(status_code=404, detail="Score not available for this member")

    return score
