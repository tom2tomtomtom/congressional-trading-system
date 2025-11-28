"""
Analysis API router - Statistical analysis, leaderboards, and insights.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import get_api_key, require_api_key, require_journalist, APIKeyData
from ..schemas import (
    Leaderboard,
    AggregateStats,
    StoryFormat,
    StoryResponse,
    StoryGenerateRequest,
    AccessTierEnum,
)
from ..services import AnalysisService, StoryService

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get(
    "/leaderboard",
    response_model=Leaderboard,
    summary="Get rankings leaderboard",
    description="""
    Get leaderboard rankings of congressional members by various metrics.

    **Available Categories:**
    - `conviction`: Ranked by average conviction score on trades
    - `volume`: Ranked by total trade volume
    - `activity`: Ranked by number of trades

    Returns the top 20 members in each category.
    """,
)
async def get_leaderboard(
    category: str = Query(
        "conviction",
        description="Ranking category (conviction, volume, activity)",
        regex="^(conviction|volume|activity)$",
    ),
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> Leaderboard:
    """Get leaderboard rankings."""
    return AnalysisService.get_leaderboard(category)


@router.get(
    "/stats",
    response_model=AggregateStats,
    summary="Get aggregate statistics",
    description="""
    Get aggregate statistics about congressional trading.

    Includes:
    - Total members and trades tracked
    - Total trade volume
    - Breakdown by party and chamber
    - Breakdown by transaction type
    - Average conviction score
    - Count of high-conviction trades
    - Late filing count
    """,
)
async def get_stats(
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> AggregateStats:
    """Get aggregate statistics."""
    return AnalysisService.get_stats()


@router.post(
    "/stories/generate",
    response_model=StoryResponse,
    summary="Generate story",
    description="""
    Generate an AI-powered narrative about a trade or member.

    **Requires Journalist tier or higher.**

    **Story Formats:**
    - `tweet`: Single tweet (280 characters)
    - `thread`: Twitter thread (3-5 tweets)
    - `news_brief`: 200-word news brief
    - `deep_dive`: Full investigation narrative
    - `data_card`: Key stats for social sharing

    Either `trade_id` or `member_id` must be provided.
    """,
    tags=["stories"],
    responses={
        200: {"description": "Story generated successfully"},
        400: {"description": "Invalid request - must provide trade_id or member_id"},
        403: {"description": "Insufficient access tier"},
        404: {"description": "Trade or member not found"},
    },
)
async def generate_story(
    request: StoryGenerateRequest,
    api_key: APIKeyData = Depends(require_api_key),
) -> StoryResponse:
    """Generate a story about a trade or member."""
    # Check tier
    allowed_tiers = [AccessTierEnum.JOURNALIST, AccessTierEnum.ACADEMIC, AccessTierEnum.PREMIUM]
    if api_key.tier not in allowed_tiers:
        raise HTTPException(
            status_code=403,
            detail="Story generation requires Journalist, Academic, or Premium tier access",
        )

    if not request.trade_id and not request.member_id:
        raise HTTPException(
            status_code=400,
            detail="Either trade_id or member_id must be provided",
        )

    story = StoryService.generate_story(
        trade_id=request.trade_id,
        member_id=request.member_id,
        format=request.format,
    )

    if not story:
        raise HTTPException(
            status_code=404,
            detail="Trade or member not found",
        )

    return story


@router.get(
    "/hall-of-shame",
    response_model=Leaderboard,
    summary="Get Hall of Shame",
    description="""
    Get the "Hall of Shame" - members with the highest suspicion scores.

    Returns the top 20 members by average conviction score on their trades.
    Higher scores indicate more suspicious trading patterns.
    """,
)
async def get_hall_of_shame(
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> Leaderboard:
    """Get Hall of Shame leaderboard."""
    leaderboard = AnalysisService.get_leaderboard("conviction")
    leaderboard.category = "hall_of_shame"
    leaderboard.description = "Members with the most suspicious trading patterns"
    return leaderboard


@router.get(
    "/hall-of-fame",
    response_model=Leaderboard,
    summary="Get Hall of Fame",
    description="""
    Get the "Hall of Fame" - members with the lowest suspicion scores.

    Returns members who demonstrate transparent, ethical trading behavior.
    This includes members who:
    - Don't trade individual stocks
    - File disclosures promptly
    - Avoid trading in sectors they oversee
    """,
)
async def get_hall_of_fame(
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> Leaderboard:
    """Get Hall of Fame leaderboard."""
    leaderboard = AnalysisService.get_leaderboard("conviction")
    # Reverse the order for Hall of Fame (lowest scores = best)
    leaderboard.entries.reverse()
    for i, entry in enumerate(leaderboard.entries):
        entry.rank = i + 1
    leaderboard.category = "hall_of_fame"
    leaderboard.description = "Members with the most transparent trading behavior"
    return leaderboard
