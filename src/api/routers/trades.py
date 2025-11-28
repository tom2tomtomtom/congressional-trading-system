"""
Trades API router - Stock trade disclosures and analysis.
"""

from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import get_api_key, require_api_key, APIKeyData
from ..schemas import (
    TradeListResponse,
    TradeDetail,
    TradeSummary,
    ConvictionAnalysis,
    TimingAnalysis,
    TransactionTypeEnum,
    AssetTypeEnum,
)
from ..services import TradeService, AnalysisService

router = APIRouter(prefix="/trades", tags=["trades"])


@router.get(
    "",
    response_model=TradeListResponse,
    summary="List all trades",
    description="""
    Get a paginated list of all disclosed stock trades.

    Supports filtering by member, symbol, transaction type, amount range, and date range.
    Results are sorted by transaction date in descending order by default.
    """,
)
async def list_trades(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    member_id: Optional[int] = Query(None, description="Filter by member ID"),
    symbol: Optional[str] = Query(None, description="Filter by stock symbol"),
    transaction_type: Optional[str] = Query(
        None, description="Filter by type (purchase, sale, exchange)"
    ),
    min_amount: Optional[float] = Query(
        None, ge=0, description="Minimum trade amount"
    ),
    max_amount: Optional[float] = Query(
        None, ge=0, description="Maximum trade amount"
    ),
    start_date: Optional[date] = Query(
        None, description="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[date] = Query(
        None, description="End date (YYYY-MM-DD)"
    ),
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> TradeListResponse:
    """List all trades with optional filtering."""
    trades, total = TradeService.get_trades(
        page=page,
        per_page=per_page,
        member_id=member_id,
        symbol=symbol,
        transaction_type=transaction_type,
        min_amount=min_amount,
        max_amount=max_amount,
        start_date=start_date,
        end_date=end_date,
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
    "/{trade_id}",
    response_model=TradeDetail,
    summary="Get trade details",
    description="""
    Get detailed information about a specific trade.

    Includes full transaction details, filing information, and basic analysis flags.
    """,
)
async def get_trade(
    trade_id: int,
    api_key: Optional[APIKeyData] = Depends(get_api_key),
) -> TradeDetail:
    """Get detailed information about a trade."""
    trade = TradeService.get_trade(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.get(
    "/{trade_id}/conviction",
    response_model=ConvictionAnalysis,
    summary="Get conviction analysis",
    description="""
    Get AI-powered conviction score analysis for a trade.

    The conviction score (0-100) indicates how likely a trade appears to be
    based on non-public information. Factors include:

    - **Committee Access** (25 pts): Member's committee oversight of the traded sector
    - **Timing Proximity** (25 pts): Closeness to market-moving events
    - **Filing Delay** (15 pts): Days between trade and disclosure
    - **Trade Size Anomaly** (15 pts): Size relative to member's typical trades
    - **Historical Pattern** (10 pts): Deviation from normal trading behavior
    - **Sector Concentration** (10 pts): Over-weighting in oversight sectors

    **Risk Levels:**
    - 0-30: Low (green)
    - 30-60: Medium (yellow)
    - 60-80: High (orange)
    - 80-100: Critical (red)
    """,
    responses={
        200: {"description": "Conviction analysis retrieved successfully"},
        404: {"description": "Trade not found"},
    },
)
async def get_conviction_analysis(
    trade_id: int,
    api_key: APIKeyData = Depends(require_api_key),
) -> ConvictionAnalysis:
    """Get conviction score analysis for a trade."""
    analysis = AnalysisService.get_conviction_analysis(trade_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Trade not found")
    return analysis


@router.get(
    "/{trade_id}/timing",
    response_model=TimingAnalysis,
    summary="Get timing analysis",
    description="""
    Get timing correlation analysis for a trade.

    Identifies events that occurred close to the trade date that the member
    may have had advance knowledge of:

    - Committee hearings
    - Bill introductions/votes
    - Classified briefings
    - Earnings announcements
    - FDA decisions
    - Regulatory announcements
    - Defense contracts

    Each event includes:
    - Event type and description
    - Days before/after the trade
    - Member's access level (direct, committee, public)
    - Relevance score (0-1)
    """,
    responses={
        200: {"description": "Timing analysis retrieved successfully"},
        404: {"description": "Trade not found"},
    },
)
async def get_timing_analysis(
    trade_id: int,
    api_key: APIKeyData = Depends(require_api_key),
) -> TimingAnalysis:
    """Get timing analysis for a trade."""
    analysis = AnalysisService.get_timing_analysis(trade_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Trade not found")
    return analysis
