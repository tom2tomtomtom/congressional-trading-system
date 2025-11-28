# Congressional Trading Intelligence System - Machine Learning Models
# Phase 2: Intelligence & Analytics ML Components

"""
ML Models Package

This package contains machine learning models and AI services for the
Congressional Trading Intelligence System.

Modules:
- llm_service: Claude API integration for story generation (Track F1)
- nl_query: Natural language query parsing (Track F2)
- trade_predictor: Trade prediction models
- anomaly_detector: Anomaly detection for suspicious patterns
- prediction_engine: General prediction utilities
- transformer_predictor: Transformer-based predictions
"""

# LLM Service exports (Track F1)
from .llm_service import (
    LLMService,
    LLMProvider,
    ClaudeProvider,
    MockLLMProvider,
    Story,
    StoryFormat,
    StoryType,
    PromptTemplate,
)

# Natural Language Query exports (Track F2)
from .nl_query import (
    NLQueryParser,
    NLQueryService,
    ParsedQuery,
    QueryFilter,
    QueryIntent,
    FilterOperator,
)

__all__ = [
    # LLM Service (F1)
    "LLMService",
    "LLMProvider",
    "ClaudeProvider",
    "MockLLMProvider",
    "Story",
    "StoryFormat",
    "StoryType",
    "PromptTemplate",
    # NL Query (F2)
    "NLQueryParser",
    "NLQueryService",
    "ParsedQuery",
    "QueryFilter",
    "QueryIntent",
    "FilterOperator",
]