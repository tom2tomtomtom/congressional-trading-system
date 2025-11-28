"""
Congressional Trading Intelligence System - Notifications Package

Track D - Task D3: Email Alert System

Provides email notification capabilities for trade alerts and digests.
"""

from .email_service import (
    EmailService,
    EmailCredentials,
    EmailTemplate,
    Subscriber,
    SubscriptionManager,
)

__all__ = [
    "EmailService",
    "EmailCredentials",
    "EmailTemplate",
    "Subscriber",
    "SubscriptionManager",
]
