#!/usr/bin/env python3
"""
Test script for new data sources
"""

import os
import sys
sys.path.insert(0, 'src')

from data_sources.congress_gov_scraper import CongressGovClient
from data_sources.house_disclosure_scraper import HouseDisclosureScraper

def test_congress_gov():
    """Test Congress.gov API"""
    print("🏛️  Testing Congress.gov API")
    print("=" * 40)
    
    # Test without API key first
    client = CongressGovClient()
    
    # Get a few members to test
    members = client.get_members(congress=118, chamber='house')
    print(f"✅ Retrieved {len(members)} House members")
    
    if members:
        sample = members[0]
        print(f"Sample: {sample['full_name']} ({sample['party']}-{sample['state']})")
    
    # Get committees
    committees = client.get_committees(congress=118)
    print(f"✅ Retrieved {len(committees)} committees")
    
    return len(members) > 0 and len(committees) > 0

def test_house_scraper():
    """Test House disclosure scraper"""
    print("\n🏛️  Testing House Disclosure Scraper")
    print("=" * 40)
    
    scraper = HouseDisclosureScraper(delay_seconds=0.5)  # Faster for testing
    
    # Get available years
    years = scraper.get_available_years()
    print(f"✅ Available years: {years[:3]}...")  # Show first 3
    
    # Test search for recent year
    if years:
        recent_disclosures = scraper.search_disclosures(year=years[0])
        print(f"✅ Found {len(recent_disclosures)} disclosures for {years[0]}")
        
        if recent_disclosures:
            sample = recent_disclosures[0]
            print(f"Sample: {sample['full_name']} - {sample['report_type']}")
    
    return len(years) > 0

def test_combined_data():
    """Test getting combined data from both sources"""
    print("\n🔄 Testing Combined Data Collection")
    print("=" * 40)
    
    # Congress.gov data
    congress_client = CongressGovClient()
    congress_data = congress_client.get_comprehensive_data()
    
    # House disclosure data  
    house_scraper = HouseDisclosureScraper()
    house_data = house_scraper.get_comprehensive_data()
    
    print(f"✅ Congress.gov: {congress_data['total_members']} members, {congress_data['total_committees']} committees")
    print(f"✅ House disclosures: {house_data['total_members']} members, {house_data['total_trades']} trades")
    
    # Show data structure
    if congress_data['members']:
        member = congress_data['members'][0]
        print(f"\nSample member data structure:")
        print(f"  Name: {member.get('full_name')}")
        print(f"  Party: {member.get('party')}")
        print(f"  State: {member.get('state')}")
        print(f"  Chamber: {member.get('chamber')}")
        print(f"  Active: {member.get('is_active')}")
    
    return True

def main():
    """Run all tests"""
    print("🧪 Congressional Data Sources Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Test individual sources
    try:
        results['congress_gov'] = test_congress_gov()
    except Exception as e:
        print(f"❌ Congress.gov test failed: {e}")
        results['congress_gov'] = False
    
    try:
        results['house_scraper'] = test_house_scraper()
    except Exception as e:
        print(f"❌ House scraper test failed: {e}")
        results['house_scraper'] = False
    
    try:
        results['combined'] = test_combined_data()
    except Exception as e:
        print(f"❌ Combined data test failed: {e}")
        results['combined'] = False
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ You have working data sources for congressional trading intelligence")
        print(f"\nNext steps:")
        print(f"1. Get free Congress.gov API key: https://api.congress.gov/sign-up/")
        print(f"2. Add to .env: CONGRESS_GOV_API_KEY=your_key_here")
        print(f"3. Run the full application!")
    else:
        print(f"\n⚠️  Some tests failed - check network connection and try again")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)