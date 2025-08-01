"""
Congressional member and committee models
"""

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Enum, ForeignKey,
    Integer, String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, Mapped
import enum

from .base import Base, TimestampMixin, IDMixin


class Chamber(enum.Enum):
    """Congressional chamber enumeration"""
    HOUSE = "house"
    SENATE = "senate"


class Party(enum.Enum):
    """Political party enumeration"""
    DEMOCRAT = "D"
    REPUBLICAN = "R"
    INDEPENDENT = "I"
    OTHER = "O"


class CommitteeType(enum.Enum):
    """Committee type enumeration"""
    STANDING = "standing"
    SELECT = "select"
    JOINT = "joint"
    SUBCOMMITTEE = "subcommittee"


class Member(Base, TimestampMixin, IDMixin):
    """Congressional member model"""
    
    __tablename__ = "members"
    
    # Basic Information
    bioguide_id = Column(String(20), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False, index=True)
    
    # Political Information
    party = Column(Enum(Party), nullable=False, index=True)
    state = Column(String(2), nullable=False, index=True)
    chamber = Column(Enum(Chamber), nullable=False, index=True)
    district = Column(String(10), nullable=True)  # For House members
    
    # Status Information
    is_active = Column(Boolean, default=True, nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    
    # Contact Information
    website = Column(String(500), nullable=True)
    twitter_handle = Column(String(100), nullable=True)
    
    # Additional Information
    date_of_birth = Column(Date, nullable=True)
    leadership_role = Column(String(255), nullable=True)
    
    # Relationships
    trades: Mapped[List["Trade"]] = relationship(
        "Trade", back_populates="member", cascade="all, delete-orphan"
    )
    committee_memberships: Mapped[List["CommitteeMembership"]] = relationship(
        "CommitteeMembership", back_populates="member", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult", back_populates="member", cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_member_party_state", "party", "state"),
        Index("ix_member_chamber_active", "chamber", "is_active"),
        UniqueConstraint("bioguide_id", name="uq_member_bioguide"),
    )
    
    def __repr__(self) -> str:
        return f"<Member {self.full_name} ({self.party.value}-{self.state})>"
    
    @property
    def display_name(self) -> str:
        """Get formatted display name"""
        return f"{self.full_name} ({self.party.value}-{self.state})"
    
    @property
    def is_senator(self) -> bool:
        """Check if member is a senator"""
        return self.chamber == Chamber.SENATE
    
    @property
    def is_representative(self) -> bool:
        """Check if member is a representative"""
        return self.chamber == Chamber.HOUSE


class Committee(Base, TimestampMixin, IDMixin):
    """Congressional committee model"""
    
    __tablename__ = "committees"
    
    # Basic Information
    name = Column(String(500), nullable=False)
    abbreviation = Column(String(50), nullable=True)
    committee_type = Column(Enum(CommitteeType), nullable=False)
    chamber = Column(Enum(Chamber), nullable=True)  # Some committees are joint
    
    # Metadata
    description = Column(Text, nullable=True)
    jurisdiction = Column(Text, nullable=True)
    website = Column(String(500), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Parent committee (for subcommittees)
    parent_committee_id = Column(
        Integer, ForeignKey("committees.id"), nullable=True
    )
    
    # Relationships
    parent_committee: Mapped[Optional["Committee"]] = relationship(
        "Committee", remote_side=[id], back_populates="subcommittees"
    )
    subcommittees: Mapped[List["Committee"]] = relationship(
        "Committee", back_populates="parent_committee", cascade="all, delete-orphan"
    )
    memberships: Mapped[List["CommitteeMembership"]] = relationship(
        "CommitteeMembership", back_populates="committee", cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_committee_name", "name"),
        Index("ix_committee_type_chamber", "committee_type", "chamber"),
        Index("ix_committee_active", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<Committee {self.name}>"
    
    @property
    def full_name(self) -> str:
        """Get full committee name including parent if subcommittee"""
        if self.parent_committee:
            return f"{self.parent_committee.name} - {self.name}"
        return self.name


class CommitteeMembership(Base, TimestampMixin, IDMixin):
    """Congressional committee membership model"""
    
    __tablename__ = "committee_memberships"
    
    # Foreign Keys
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    committee_id = Column(Integer, ForeignKey("committees.id"), nullable=False)
    
    # Membership Details
    position = Column(String(100), nullable=True)  # Chair, Ranking Member, etc.
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    member: Mapped["Member"] = relationship("Member", back_populates="committee_memberships")
    committee: Mapped["Committee"] = relationship("Committee", back_populates="memberships")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "member_id", "committee_id", "start_date",
            name="uq_committee_membership"
        ),
        Index("ix_membership_member_active", "member_id", "is_active"),
        Index("ix_membership_committee_active", "committee_id", "is_active"),
    )
    
    def __repr__(self) -> str:
        position_str = f" ({self.position})" if self.position else ""
        return f"<Membership {self.member.full_name} -> {self.committee.name}{position_str}>"
    
    @property
    def is_leadership_position(self) -> bool:
        """Check if this is a leadership position"""
        if not self.position:
            return False
        leadership_terms = ["chair", "ranking", "leader", "whip"]
        return any(term in self.position.lower() for term in leadership_terms)