"""
Base database models and mixins for the Congressional Trading Intelligence System
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Integer, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps"""
    
    created_at = Column(
        DateTime,
        nullable=False,
        default=func.now(),
        server_default=func.now()
    )
    
    updated_at = Column(
        DateTime,
        nullable=False,
        default=func.now(),
        server_default=func.now(),
        onupdate=func.now()
    )


class IDMixin:
    """Mixin to add auto-incrementing ID primary key"""
    
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False
    )