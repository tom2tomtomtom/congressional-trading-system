#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - ProPublica Congress API Client
Integration for congressional member data, voting records, and committee information.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ProPublicaMember:
    """Data model for ProPublica member information."""
    id: str
    title: str
    short_title: str
    api_uri: str
    first_name: str
    middle_name: Optional[str]
    last_name: str
    suffix: Optional[str]
    date_of_birth: str
    gender: str
    party: str
    leadership_role: Optional[str]
    twitter_account: Optional[str]
    facebook_account: Optional[str]
    youtube_account: Optional[str]
    govtrack_id: str
    cspan_id: str
    votesmart_id: str
    icpsr_id: str
    crp_id: str
    google_entity_id: str
    fec_candidate_id: str
    url: str
    rss_url: str
    contact_form: Optional[str]
    in_office: bool
    cook_pvi: Optional[str]
    dw_nominate: Optional[float]
    ideal_point: Optional[float]
    seniority: str
    next_election: str
    total_votes: int
    missed_votes: int
    total_present: int
    last_updated: str
    ocd_id: str
    office: str
    phone: str
    fax: Optional[str]
    state: str
    senate_class: Optional[str]
    state_rank: Optional[str]
    lis_id: Optional[str]
    missed_votes_pct: float
    votes_with_party_pct: float
    votes_against_party_pct: float

@dataclass  
class VotingRecord:
    """Data model for voting record information."""
    member_id: str
    total_votes: int
    missed_votes: int
    missed_votes_pct: float
    votes_with_party_pct: float
    votes_against_party_pct: float

@dataclass
class CommitteeMembership:
    """Data model for committee membership."""
    name: str
    code: str
    api_uri: str
    side: str
    title: str
    rank_in_party: int
    begin_date: str
    end_date: Optional[str]

class ProPublicaAPIClient:
    """
    Client for accessing ProPublica Congress API data.
    Handles authentication, rate limiting, and data parsing.
    """
    
    BASE_URL = "https://api.propublica.org/congress/v1"
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize ProPublica API client."""
        self.api_key = api_key or os.getenv('PROPUBLICA_API_KEY')
        
        if not self.api_key:
            raise ValueError("ProPublica API key is required")
        
        self.config = self._load_config(config_path)
        self.session = self._setup_session()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get('rate_limit', 5000),
            time_window=86400  # 24 hours
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('api_sources', {}).get('propublica', {})
        
        return {
            'base_url': self.BASE_URL,
            'rate_limit': 5000,
            'timeout': 30,
            'retry_attempts': 3
        }
    
    def _setup_session(self) -> requests.Session:
        """Set up requests session with retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.get('retry_attempts', 3),
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'User-Agent': 'Congressional-Trading-Intelligence-System/1.0',
            'Accept': 'application/json',
            'X-API-Key': self.api_key
        })
        
        return session
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to ProPublica API."""
        self.rate_limiter.wait_if_needed()
        
        url = urljoin(self.config['base_url'], endpoint)
        
        try:
            response = self.session.get(
                url, 
                params=params,
                timeout=self.config.get('timeout', 30)
            )
            response.raise_for_status()
            
            self.rate_limiter.record_request()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            raise
    
    def get_chamber_members(self, congress: int, chamber: str) -> List[ProPublicaMember]:
        """
        Get all members for a specific chamber and congress.
        
        Args:
            congress: Congress number (e.g., 118)
            chamber: 'house' or 'senate'
            
        Returns:
            List of ProPublicaMember objects
        """
        logger.info(f"Fetching {chamber} members for Congress {congress}")
        
        endpoint = f"/{congress}/{chamber}/members.json"
        
        try:
            response = self._make_request(endpoint)
            results = response.get('results', [])
            
            if not results:
                logger.warning(f"No results found for {chamber} in Congress {congress}")
                return []
            
            members_data = results[0].get('members', [])
            members = []
            
            for member_data in members_data:
                member = self._parse_member_data(member_data)
                if member:
                    members.append(member)
            
            logger.info(f"Successfully fetched {len(members)} {chamber} members")
            return members
            
        except Exception as e:
            logger.error(f"Error fetching {chamber} members: {e}")
            return []
    
    def _parse_member_data(self, data: Dict[str, Any]) -> Optional[ProPublicaMember]:
        """Parse member data from API response."""
        try:
            return ProPublicaMember(
                id=data.get('id', ''),
                title=data.get('title', ''),
                short_title=data.get('short_title', ''),
                api_uri=data.get('api_uri', ''),
                first_name=data.get('first_name', ''),
                middle_name=data.get('middle_name'),
                last_name=data.get('last_name', ''),
                suffix=data.get('suffix'),
                date_of_birth=data.get('date_of_birth', ''),
                gender=data.get('gender', ''),
                party=data.get('party', ''),
                leadership_role=data.get('leadership_role'),
                twitter_account=data.get('twitter_account'),
                facebook_account=data.get('facebook_account'),
                youtube_account=data.get('youtube_account'),
                govtrack_id=data.get('govtrack_id', ''),
                cspan_id=data.get('cspan_id', ''),
                votesmart_id=data.get('votesmart_id', ''),
                icpsr_id=data.get('icpsr_id', ''),
                crp_id=data.get('crp_id', ''),
                google_entity_id=data.get('google_entity_id', ''),
                fec_candidate_id=data.get('fec_candidate_id', ''),
                url=data.get('url', ''),
                rss_url=data.get('rss_url', ''),
                contact_form=data.get('contact_form'),
                in_office=data.get('in_office', True),
                cook_pvi=data.get('cook_pvi'),
                dw_nominate=data.get('dw_nominate'),
                ideal_point=data.get('ideal_point'),
                seniority=data.get('seniority', ''),
                next_election=data.get('next_election', ''),
                total_votes=data.get('total_votes', 0),
                missed_votes=data.get('missed_votes', 0),
                total_present=data.get('total_present', 0),
                last_updated=data.get('last_updated', ''),
                ocd_id=data.get('ocd_id', ''),
                office=data.get('office', ''),
                phone=data.get('phone', ''),
                fax=data.get('fax'),
                state=data.get('state', ''),
                senate_class=data.get('senate_class'),
                state_rank=data.get('state_rank'),
                lis_id=data.get('lis_id'),
                missed_votes_pct=data.get('missed_votes_pct', 0.0),
                votes_with_party_pct=data.get('votes_with_party_pct', 0.0),
                votes_against_party_pct=data.get('votes_against_party_pct', 0.0)
            )
        except Exception as e:
            logger.warning(f"Failed to parse member data: {e}")
            return None
    
    def get_member_details(self, member_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific member."""
        endpoint = f"/members/{member_id}.json"
        
        try:
            response = self._make_request(endpoint)
            results = response.get('results', [])
            
            if results:
                return results[0]
            return None
            
        except Exception as e:
            logger.error(f"Error fetching member details for {member_id}: {e}")
            return None
    
    def get_member_committee_assignments(self, member_id: str, congress: int) -> List[CommitteeMembership]:
        """Get committee assignments for a specific member."""
        endpoint = f"/members/{member_id}/committees/{congress}.json"
        
        try:
            response = self._make_request(endpoint)
            results = response.get('results', [])
            
            if not results:
                return []
            
            committees = []
            for committee_data in results[0].get('committees', []):
                committee = self._parse_committee_membership(committee_data)
                if committee:
                    committees.append(committee)
            
            return committees
            
        except Exception as e:
            logger.error(f"Error fetching committee assignments for {member_id}: {e}")
            return []
    
    def _parse_committee_membership(self, data: Dict[str, Any]) -> Optional[CommitteeMembership]:
        """Parse committee membership data."""
        try:
            return CommitteeMembership(
                name=data.get('name', ''),
                code=data.get('code', ''),
                api_uri=data.get('api_uri', ''),
                side=data.get('side', ''),
                title=data.get('title', ''),
                rank_in_party=data.get('rank_in_party', 0),
                begin_date=data.get('begin_date', ''),
                end_date=data.get('end_date')
            )
        except Exception as e:
            logger.warning(f"Failed to parse committee membership: {e}")
            return None
    
    def get_voting_record(self, member_id: str, congress: int, session: int = 1) -> Optional[VotingRecord]:
        """Get voting record for a specific member."""
        endpoint = f"/members/{member_id}/votes/{congress}/{session}.json"
        
        try:
            response = self._make_request(endpoint)
            results = response.get('results', [])
            
            if results:
                member_data = results[0]
                return VotingRecord(
                    member_id=member_id,
                    total_votes=member_data.get('total_votes', 0),
                    missed_votes=member_data.get('missed_votes', 0),
                    missed_votes_pct=member_data.get('missed_votes_pct', 0.0),
                    votes_with_party_pct=member_data.get('votes_with_party_pct', 0.0),
                    votes_against_party_pct=member_data.get('votes_against_party_pct', 0.0)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching voting record for {member_id}: {e}")
            return None
    
    def get_recent_bills(self, congress: int, chamber: str, bill_type: str = "introduced") -> List[Dict[str, Any]]:
        """
        Get recent bills for a chamber.
        
        Args:
            congress: Congress number
            chamber: 'house' or 'senate'  
            bill_type: 'introduced', 'updated', 'active', 'passed', 'enacted', 'vetoed'
        """
        endpoint = f"/{congress}/{chamber}/bills/{bill_type}.json"
        
        try:
            response = self._make_request(endpoint)
            results = response.get('results', [])
            
            if results:
                return results[0].get('bills', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching recent bills: {e}")
            return []
    
    def get_committee_hearings(self, congress: int, chamber: str) -> List[Dict[str, Any]]:
        """Get committee hearings for a chamber."""
        endpoint = f"/{congress}/{chamber}/committees/hearings.json"
        
        try:
            response = self._make_request(endpoint)
            results = response.get('results', [])
            
            if results:
                return results[0].get('hearings', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching committee hearings: {e}")
            return []
    
    def get_member_statements(self, member_id: str, congress: int, offset: int = 0) -> List[Dict[str, Any]]:
        """Get statements for a specific member."""
        endpoint = f"/members/{member_id}/statements/{congress}.json"
        params = {'offset': offset}
        
        try:
            response = self._make_request(endpoint, params)
            results = response.get('results', [])
            
            if results:
                return results[0].get('statements', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching statements for {member_id}: {e}")
            return []
    
    def get_all_members(self, congress: int = 118) -> Iterator[ProPublicaMember]:
        """Get all congressional members from both chambers."""
        logger.info(f"Fetching all members for Congress {congress}")
        
        # Get House members
        house_members = self.get_chamber_members(congress, "house")
        for member in house_members:
            yield member
        
        # Get Senate members
        senate_members = self.get_chamber_members(congress, "senate")
        for member in senate_members:
            yield member

class RateLimiter:
    """Rate limiter for ProPublica API (5000 requests per day)."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # Check if we're at the limit
        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request is outside the window
            wait_time = self.time_window - (now - self.requests[0]) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
    
    def record_request(self):
        """Record a new request."""
        self.requests.append(time.time())

def main():
    """Test function for ProPublica API client."""
    logging.basicConfig(level=logging.INFO)
    
    client = ProPublicaAPIClient()
    
    # Test fetching members
    print("Fetching congressional members...")
    member_count = 0
    for member in client.get_all_members(118):
        member_count += 1
        if member_count <= 5:  # Show first 5 members
            print(f"  {member.first_name} {member.last_name} ({member.party}-{member.state})")
            print(f"    Votes with party: {member.votes_with_party_pct}%")
            print(f"    Missed votes: {member.missed_votes_pct}%")
        if member_count >= 10:  # Limit for testing
            break
    
    print(f"Found {member_count} members")

if __name__ == "__main__":
    main()