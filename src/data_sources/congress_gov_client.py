#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Congress.gov API Client
Official API integration for congressional member data, committee information, and legislation.
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
class CongressionalMember:
    """Data model for congressional member information."""
    bioguide_id: str
    first_name: str
    last_name: str
    full_name: str
    party: str
    state: str
    district: Optional[int]
    chamber: str
    served_from: str
    served_to: Optional[str]
    birth_date: Optional[str]
    official_full_name: Optional[str]
    nickname: Optional[str]

@dataclass
class Committee:
    """Data model for congressional committee information."""
    thomas_id: str
    name: str
    chamber: str
    committee_type: str
    parent_committee_id: Optional[str]
    jurisdiction: Optional[str]
    website_url: Optional[str]

@dataclass
class Bill:
    """Data model for congressional bill information."""
    bill_id: str
    title: str
    bill_type: str
    number: int
    congress: int
    introduced_date: Optional[str]
    latest_action_date: Optional[str]
    latest_action: Optional[str]
    status: Optional[str]
    summary: Optional[str]
    policy_area: Optional[str]
    subjects: List[str]
    sponsor_bioguide_id: Optional[str]

class CongressGovAPIClient:
    """
    Client for accessing Congress.gov API data.
    Handles rate limiting, authentication, and data parsing.
    """
    
    BASE_URL = "https://api.congress.gov/v3"
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize Congress.gov API client."""
        self.api_key = api_key or os.getenv('CONGRESS_API_KEY')
        
        if not self.api_key:
            raise ValueError("Congress.gov API key is required")
        
        self.config = self._load_config(config_path)
        self.session = self._setup_session()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get('rate_limit', 5000),
            time_window=3600  # 1 hour
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('api_sources', {}).get('congress_gov', {})
        
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
        """Make authenticated request to Congress.gov API."""
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
    
    def get_all_members(self, congress: int = 118) -> Iterator[CongressionalMember]:
        """
        Get all congressional members for a specific congress.
        
        Args:
            congress: Congress number (e.g., 118 for 2023-2024)
            
        Yields:
            CongressionalMember objects
        """
        logger.info(f"Fetching all members for Congress {congress}")
        
        # Get House members
        for member in self._get_chamber_members(congress, "house"):
            yield member
        
        # Get Senate members  
        for member in self._get_chamber_members(congress, "senate"):
            yield member
    
    def _get_chamber_members(self, congress: int, chamber: str) -> Iterator[CongressionalMember]:
        """Get members for a specific chamber."""
        endpoint = f"/congress/{congress}/{chamber}/members"
        offset = 0
        limit = 250
        
        while True:
            params = {'offset': offset, 'limit': limit, 'format': 'json'}
            
            try:
                response = self._make_request(endpoint, params)
                members_data = response.get('members', [])
                
                if not members_data:
                    break
                
                for member_data in members_data:
                    member = self._parse_member_data(member_data, chamber)
                    if member:
                        yield member
                
                # Check if we have more pages
                if len(members_data) < limit:
                    break
                
                offset += limit
                
            except Exception as e:
                logger.error(f"Error fetching {chamber} members at offset {offset}: {e}")
                break
    
    def _parse_member_data(self, data: Dict[str, Any], chamber: str) -> Optional[CongressionalMember]:
        """Parse member data from API response."""
        try:
            return CongressionalMember(
                bioguide_id=data.get('bioguideId', ''),
                first_name=data.get('firstName', ''),
                last_name=data.get('lastName', ''),
                full_name=f"{data.get('firstName', '')} {data.get('lastName', '')}".strip(),
                party=data.get('party', ''),
                state=data.get('state', ''),
                district=data.get('district') if chamber == 'house' else None,
                chamber=chamber.title(),
                served_from=data.get('startDate', ''),
                served_to=data.get('endDate'),
                birth_date=data.get('birthDate'),
                official_full_name=data.get('officialName'),
                nickname=data.get('nickname')
            )
        except Exception as e:
            logger.warning(f"Failed to parse member data: {e}")
            return None
    
    def get_member_details(self, bioguide_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific member."""
        endpoint = f"/member/{bioguide_id}"
        
        try:
            response = self._make_request(endpoint)
            return response.get('member', {})
        except Exception as e:
            logger.error(f"Error fetching member details for {bioguide_id}: {e}")
            return None
    
    def get_all_committees(self, congress: int = 118) -> Iterator[Committee]:
        """Get all congressional committees."""
        logger.info(f"Fetching all committees for Congress {congress}")
        
        for chamber in ['house', 'senate', 'joint']:
            for committee in self._get_chamber_committees(congress, chamber):
                yield committee
    
    def _get_chamber_committees(self, congress: int, chamber: str) -> Iterator[Committee]:
        """Get committees for a specific chamber."""
        endpoint = f"/congress/{congress}/{chamber}/committees"
        
        try:
            response = self._make_request(endpoint)
            committees_data = response.get('committees', [])
            
            for committee_data in committees_data:
                committee = self._parse_committee_data(committee_data, chamber)
                if committee:
                    yield committee
                    
                # Get subcommittees
                for subcommittee in self._get_subcommittees(congress, chamber, committee_data.get('systemCode')):
                    yield subcommittee
                    
        except Exception as e:
            logger.error(f"Error fetching {chamber} committees: {e}")
    
    def _get_subcommittees(self, congress: int, chamber: str, parent_code: str) -> Iterator[Committee]:
        """Get subcommittees for a parent committee."""
        endpoint = f"/congress/{congress}/{chamber}/committees/{parent_code}/subcommittees"
        
        try:
            response = self._make_request(endpoint)
            subcommittees_data = response.get('subcommittees', [])
            
            for subcommittee_data in subcommittees_data:
                subcommittee = self._parse_committee_data(subcommittee_data, chamber, parent_code)
                if subcommittee:
                    yield subcommittee
                    
        except Exception as e:
            logger.warning(f"Error fetching subcommittees for {parent_code}: {e}")
    
    def _parse_committee_data(self, data: Dict[str, Any], chamber: str, parent_code: Optional[str] = None) -> Optional[Committee]:
        """Parse committee data from API response."""
        try:
            return Committee(
                thomas_id=data.get('systemCode', ''),
                name=data.get('name', ''),
                chamber=chamber.title(),
                committee_type='Subcommittee' if parent_code else 'Standing',
                parent_committee_id=parent_code,
                jurisdiction=None,  # Would need separate API call
                website_url=data.get('url')
            )
        except Exception as e:
            logger.warning(f"Failed to parse committee data: {e}")
            return None
    
    def get_committee_members(self, congress: int, chamber: str, committee_code: str) -> List[Dict[str, Any]]:
        """Get members of a specific committee."""
        endpoint = f"/congress/{congress}/{chamber}/committees/{committee_code}/members"
        
        try:
            response = self._make_request(endpoint)
            return response.get('members', [])
        except Exception as e:
            logger.error(f"Error fetching committee members for {committee_code}: {e}")
            return []
    
    def get_recent_bills(self, congress: int = 118, limit: int = 250) -> Iterator[Bill]:
        """Get recently introduced bills."""
        logger.info(f"Fetching recent bills for Congress {congress}")
        
        endpoint = f"/congress/{congress}/bills"
        offset = 0
        
        while True:
            params = {'offset': offset, 'limit': limit, 'format': 'json'}
            
            try:
                response = self._make_request(endpoint, params)
                bills_data = response.get('bills', [])
                
                if not bills_data:
                    break
                
                for bill_data in bills_data:
                    bill = self._parse_bill_data(bill_data)
                    if bill:
                        yield bill
                
                if len(bills_data) < limit:
                    break
                
                offset += limit
                
            except Exception as e:
                logger.error(f"Error fetching bills at offset {offset}: {e}")
                break
    
    def _parse_bill_data(self, data: Dict[str, Any]) -> Optional[Bill]:
        """Parse bill data from API response."""
        try:
            return Bill(
                bill_id=f"{data.get('type', '').lower()}{data.get('number', '')}-{data.get('congress', '')}",
                title=data.get('title', ''),
                bill_type=data.get('type', ''),
                number=data.get('number', 0),
                congress=data.get('congress', 0),
                introduced_date=data.get('introducedDate'),
                latest_action_date=data.get('latestAction', {}).get('actionDate'),
                latest_action=data.get('latestAction', {}).get('text'),
                status=None,  # Would need separate API call
                summary=None,  # Would need separate API call
                policy_area=data.get('policyArea', {}).get('name'),
                subjects=[],  # Would need separate API call
                sponsor_bioguide_id=data.get('sponsors', [{}])[0].get('bioguideId') if data.get('sponsors') else None
            )
        except Exception as e:
            logger.warning(f"Failed to parse bill data: {e}")
            return None

class RateLimiter:
    """Simple rate limiter for API requests."""
    
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
    """Test function for Congress.gov API client."""
    logging.basicConfig(level=logging.INFO)
    
    client = CongressGovAPIClient()
    
    # Test fetching members
    print("Fetching congressional members...")
    member_count = 0
    for member in client.get_all_members(118):
        member_count += 1
        if member_count <= 5:  # Show first 5 members
            print(f"  {member.full_name} ({member.party}-{member.state}) - {member.chamber}")
        if member_count >= 10:  # Limit for testing
            break
    
    print(f"Found {member_count} members")
    
    # Test fetching committees
    print("\nFetching committees...")
    committee_count = 0
    for committee in client.get_all_committees(118):
        committee_count += 1
        if committee_count <= 5:  # Show first 5 committees
            print(f"  {committee.name} ({committee.chamber})")
        if committee_count >= 10:  # Limit for testing
            break
    
    print(f"Found {committee_count} committees")

if __name__ == "__main__":
    main()