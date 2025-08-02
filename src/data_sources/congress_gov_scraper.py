#!/usr/bin/env python3
"""
Congress.gov Data Collector
Official API access to congressional member and committee data
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import time

logger = logging.getLogger(__name__)


class CongressGovClient:
    """
    Official Congress.gov API client
    Free API with comprehensive congressional data
    """
    
    BASE_URL = "https://api.congress.gov/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'X-API-Key': self.api_key
            })
        
        self.session.headers.update({
            'User-Agent': 'Congressional Trading Intelligence System',
            'Accept': 'application/json'
        })
    
    def get_members(self, congress: int = 118, chamber: str = None) -> List[Dict[str, Any]]:
        """
        Get congressional members
        
        Args:
            congress: Congress number (118 = current)
            chamber: 'house', 'senate', or None for both
            
        Returns:
            List of member data
        """
        try:
            url = f"{self.BASE_URL}/member/{congress}"
            params = {'format': 'json', 'limit': 250}
            
            if chamber:
                params['chamber'] = chamber
            
            logger.info(f"Fetching members for Congress {congress}, chamber: {chamber or 'both'}")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            members = data.get('members', [])
            
            # Process member data
            processed_members = []
            for member in members:
                processed_member = self._process_member_data(member)
                if processed_member:
                    processed_members.append(processed_member)
            
            logger.info(f"Retrieved {len(processed_members)} members")
            return processed_members
            
        except Exception as e:
            logger.error(f"Error fetching members: {e}")
            return []
    
    def get_member_details(self, bioguide_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific member"""
        try:
            url = f"{self.BASE_URL}/member/{bioguide_id}"
            params = {'format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('member', {})
            
        except Exception as e:
            logger.error(f"Error fetching member details for {bioguide_id}: {e}")
            return {}
    
    def get_committees(self, congress: int = 118, chamber: str = None) -> List[Dict[str, Any]]:
        """Get congressional committees"""
        try:
            url = f"{self.BASE_URL}/committee/{congress}"
            params = {'format': 'json', 'limit': 250}
            
            if chamber:
                params['chamber'] = chamber
            
            logger.info(f"Fetching committees for Congress {congress}")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            committees = data.get('committees', [])
            
            # Process committee data
            processed_committees = []
            for committee in committees:
                processed_committee = self._process_committee_data(committee)
                if processed_committee:
                    processed_committees.append(processed_committee)
            
            logger.info(f"Retrieved {len(processed_committees)} committees")
            return processed_committees
            
        except Exception as e:
            logger.error(f"Error fetching committees: {e}")
            return []
    
    def get_committee_members(self, committee_code: str, congress: int = 118) -> List[Dict[str, Any]]:
        """Get members of a specific committee"""
        try:
            url = f"{self.BASE_URL}/committee/{congress}/{committee_code}"
            params = {'format': 'json'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            committee_data = data.get('committee', {})
            
            # Extract member information
            members = []
            
            # Check for current members
            if 'members' in committee_data:
                for member in committee_data['members']:
                    processed_member = {
                        'bioguide_id': member.get('bioguideId', ''),
                        'name': member.get('name', ''),
                        'party': member.get('party', ''),
                        'state': member.get('state', ''),
                        'role': member.get('role', 'Member'),
                        'committee_code': committee_code
                    }
                    members.append(processed_member)
            
            return members
            
        except Exception as e:
            logger.error(f"Error fetching committee members for {committee_code}: {e}")
            return []
    
    def _process_member_data(self, raw_member: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw member data into standardized format"""
        try:
            # Extract name information
            name = raw_member.get('name', '')
            bioguide_id = raw_member.get('bioguideId', '')
            
            # Parse party
            party_raw = raw_member.get('partyName', '')
            party = self._normalize_party(party_raw)
            
            # Extract state and district
            state = raw_member.get('state', '')
            district = raw_member.get('district', '')
            
            # Determine chamber
            chamber = 'house' if district else 'senate'
            
            # Extract terms to get current status
            terms = raw_member.get('terms', {}).get('item', [])
            is_active = bool([term for term in terms if not term.get('endYear')])
            
            processed = {
                'bioguide_id': bioguide_id,
                'full_name': name,
                'first_name': self._extract_first_name(name),
                'last_name': self._extract_last_name(name),
                'party': party,
                'state': state,
                'chamber': chamber,
                'district': district if district else None,
                'is_active': is_active,
                'terms': terms,
                'served_from': self._get_first_term_start(terms),
                'served_to': self._get_last_term_end(terms)
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing member data: {e}")
            return {}
    
    def _process_committee_data(self, raw_committee: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw committee data into standardized format"""
        try:
            return {
                'committee_code': raw_committee.get('systemCode', ''),
                'name': raw_committee.get('name', ''),
                'chamber': raw_committee.get('chamber', '').lower(),
                'committee_type': raw_committee.get('type', {}).get('name', ''),
                'parent_committee': raw_committee.get('parent', {}).get('systemCode'),
                'is_subcommittee': bool(raw_committee.get('parent')),
                'website': raw_committee.get('url', '')
            }
            
        except Exception as e:
            logger.error(f"Error processing committee data: {e}")
            return {}
    
    def _normalize_party(self, party_raw: str) -> str:
        """Normalize party names to standard abbreviations"""
        party_mapping = {
            'Democratic': 'D',
            'Democrat': 'D',
            'Republican': 'R',
            'Independent': 'I',
            'Libertarian': 'L'
        }
        
        return party_mapping.get(party_raw, party_raw[:1].upper() if party_raw else 'U')
    
    def _extract_first_name(self, full_name: str) -> str:
        """Extract first name from full name"""
        try:
            # Handle formats like "Smith, John" or "John Smith"
            if ',' in full_name:
                parts = full_name.split(',')
                if len(parts) > 1:
                    return parts[1].strip().split()[0]
            else:
                return full_name.split()[0]
        except Exception:
            return ''
    
    def _extract_last_name(self, full_name: str) -> str:
        """Extract last name from full name"""
        try:
            if ',' in full_name:
                return full_name.split(',')[0].strip()
            else:
                return full_name.split()[-1]
        except Exception:
            return full_name
    
    def _get_first_term_start(self, terms: List[Dict[str, Any]]) -> Optional[str]:
        """Get the start date of the first term"""
        if not terms:
            return None
        
        # Sort by start year
        sorted_terms = sorted(terms, key=lambda x: x.get('startYear', 9999))
        first_term = sorted_terms[0]
        
        return f"{first_term.get('startYear')}-01-01" if first_term.get('startYear') else None
    
    def _get_last_term_end(self, terms: List[Dict[str, Any]]) -> Optional[str]:
        """Get the end date of the last term"""
        if not terms:
            return None
        
        # Find latest term with end year
        latest_end = None
        for term in terms:
            if term.get('endYear'):
                if not latest_end or term['endYear'] > latest_end:
                    latest_end = term['endYear']
        
        return f"{latest_end}-12-31" if latest_end else None
    
    def get_comprehensive_data(self) -> Dict[str, Any]:
        """Get comprehensive congressional data"""
        try:
            logger.info("Starting comprehensive Congress.gov data collection")
            
            # Get current congress members
            all_members = []
            
            # Get House members
            house_members = self.get_members(congress=118, chamber='house')
            all_members.extend(house_members)
            time.sleep(0.5)  # Rate limiting
            
            # Get Senate members
            senate_members = self.get_members(congress=118, chamber='senate')
            all_members.extend(senate_members)
            time.sleep(0.5)
            
            # Get committees
            committees = self.get_committees(congress=118)
            time.sleep(0.5)
            
            # Get committee memberships for major committees
            committee_memberships = []
            major_committees = [c for c in committees if not c.get('is_subcommittee')][:10]  # Limit for API
            
            for committee in major_committees:
                committee_code = committee.get('committee_code', '')
                if committee_code:
                    members = self.get_committee_members(committee_code)
                    for member in members:
                        committee_memberships.append({
                            'bioguide_id': member['bioguide_id'],
                            'committee_code': committee_code,
                            'committee_name': committee['name'],
                            'role': member.get('role', 'Member'),
                            'chamber': committee.get('chamber', '')
                        })
                    time.sleep(0.5)  # Rate limiting
            
            result = {
                'members': all_members,
                'committees': committees,
                'committee_memberships': committee_memberships,
                'collection_date': datetime.now().isoformat(),
                'source': 'congress_gov_api',
                'total_members': len(all_members),
                'total_committees': len(committees),
                'total_memberships': len(committee_memberships)
            }
            
            logger.info(f"Collected: {len(all_members)} members, {len(committees)} committees, "
                       f"{len(committee_memberships)} memberships")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            return {
                'members': [],
                'committees': [],
                'committee_memberships': [],
                'error': str(e)
            }


def main():
    """Test the Congress.gov client"""
    logging.basicConfig(level=logging.INFO)
    
    # Test without API key first (may have rate limits)
    client = CongressGovClient()
    
    print("🏛️  Congress.gov API Client Test")
    print("=" * 50)
    
    # Test getting members
    print("📊 Testing member retrieval...")
    members = client.get_members(congress=118, chamber='house')
    print(f"Retrieved {len(members)} House members")
    
    if members:
        sample_member = members[0]
        print(f"\nSample member: {sample_member['full_name']} ({sample_member['party']}-{sample_member['state']})")
        print(f"Chamber: {sample_member['chamber']}")
        print(f"Active: {sample_member['is_active']}")
    
    # Test getting committees
    print(f"\n📋 Testing committee retrieval...")
    committees = client.get_committees(congress=118)
    print(f"Retrieved {len(committees)} committees")
    
    if committees:
        sample_committee = committees[0]
        print(f"\nSample committee: {sample_committee['name']}")
        print(f"Chamber: {sample_committee['chamber']}")
        print(f"Type: {sample_committee['committee_type']}")
    
    # Test comprehensive data collection
    print(f"\n🔍 Testing comprehensive data collection...")
    comprehensive_data = client.get_comprehensive_data()
    
    print(f"Comprehensive data collected:")
    print(f"  - Members: {comprehensive_data['total_members']}")
    print(f"  - Committees: {comprehensive_data['total_committees']}")
    print(f"  - Memberships: {comprehensive_data['total_memberships']}")
    
    # Save to JSON
    try:
        with open('congress_gov_data.json', 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
        print(f"✅ Saved comprehensive data to congress_gov_data.json")
    except Exception as e:
        print(f"❌ Error saving data: {e}")


if __name__ == "__main__":
    main()