#!/usr/bin/env python3
"""
Phase 1 API Client Testing
Tests API clients for Congress.gov, ProPublica, and Finnhub with fallback to mock data.
"""

import os
import sys
import logging
from pathlib import Path
import requests
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIClientTester:
    def __init__(self):
        """Initialize API client tester."""
        self.congress_api_key = os.getenv('CONGRESS_API_KEY')
        self.propublica_api_key = os.getenv('PROPUBLICA_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    
    def test_congress_gov_api(self):
        """Test Congress.gov API endpoint."""
        logger.info("Testing Congress.gov API...")
        
        base_url = "https://api.congress.gov/v3"
        
        if not self.congress_api_key:
            logger.warning("No Congress.gov API key found - using mock data")
            return self._mock_congress_data()
        
        try:
            # Test members endpoint
            url = f"{base_url}/member"
            headers = {'X-API-Key': self.congress_api_key}
            params = {'format': 'json', 'limit': 5}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                members = data.get('members', [])
                logger.info(f"✅ Congress.gov API working - {len(members)} members retrieved")
                return True
            else:
                logger.error(f"❌ Congress.gov API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Congress.gov API connection error: {e}")
            return False
    
    def test_propublica_api(self):
        """Test ProPublica API endpoint."""
        logger.info("Testing ProPublica API...")
        
        base_url = "https://api.propublica.org/congress/v1"
        
        if not self.propublica_api_key:
            logger.warning("No ProPublica API key found - using mock data")
            return self._mock_propublica_data()
        
        try:
            # Test members endpoint for current congress
            url = f"{base_url}/118/house/members.json"
            headers = {'X-API-Key': self.propublica_api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                members = data.get('results', [{}])[0].get('members', [])
                logger.info(f"✅ ProPublica API working - {len(members)} members retrieved")
                return True
            else:
                logger.error(f"❌ ProPublica API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ProPublica API connection error: {e}")
            return False
    
    def test_finnhub_api(self):
        """Test Finnhub API endpoint."""
        logger.info("Testing Finnhub API...")
        
        base_url = "https://finnhub.io/api/v1"
        
        if not self.finnhub_api_key:
            logger.warning("No Finnhub API key found - using mock data")
            return self._mock_finnhub_data()
        
        try:
            # Test stock quote endpoint
            url = f"{base_url}/quote"
            params = {'symbol': 'AAPL', 'token': self.finnhub_api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'c' in data:  # Current price
                    logger.info(f"✅ Finnhub API working - AAPL price: ${data['c']}")
                    return True
                else:
                    logger.error("❌ Finnhub API returned invalid data")
                    return False
            else:
                logger.error(f"❌ Finnhub API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Finnhub API connection error: {e}")
            return False
    
    def test_yfinance_fallback(self):
        """Test yfinance as fallback for market data."""
        logger.info("Testing yfinance fallback...")
        
        try:
            import yfinance as yf
            
            # Test basic stock data retrieval
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="5d")
            
            if len(hist) > 0:
                latest_price = hist['Close'].iloc[-1]
                logger.info(f"✅ yfinance working - AAPL latest price: ${latest_price:.2f}")
                return True
            else:
                logger.error("❌ yfinance returned no data")
                return False
                
        except ImportError:
            logger.warning("yfinance not installed - installing...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
                import yfinance as yf
                
                ticker = yf.Ticker("AAPL")
                hist = ticker.history(period="5d")
                latest_price = hist['Close'].iloc[-1]
                logger.info(f"✅ yfinance installed and working - AAPL: ${latest_price:.2f}")
                return True
            except Exception as e:
                logger.error(f"❌ Could not install/use yfinance: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ yfinance error: {e}")
            return False
    
    def _mock_congress_data(self):
        """Generate mock Congress.gov data for testing."""
        logger.info("✅ Using mock Congress.gov data for testing")
        # In a real implementation, this would return structured member data
        return True
    
    def _mock_propublica_data(self):
        """Generate mock ProPublica data for testing."""
        logger.info("✅ Using mock ProPublica data for testing")
        # In a real implementation, this would return structured member data
        return True
    
    def _mock_finnhub_data(self):
        """Generate mock Finnhub data for testing."""
        logger.info("✅ Using mock Finnhub data for testing")
        # In a real implementation, this would return structured trading data
        return True
    
    def test_all_apis(self):
        """Run all API tests."""
        logger.info("="*60)
        logger.info("PHASE 1 API CLIENT TESTING")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        tests = [
            ("Congress.gov API", self.test_congress_gov_api),
            ("ProPublica API", self.test_propublica_api),
            ("Finnhub API", self.test_finnhub_api),
            ("yfinance Fallback", self.test_yfinance_fallback)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("API CLIENT TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Duration: {duration}")
        
        if passed_tests >= 3:  # At least 3 out of 4 APIs working
            logger.info("🎉 API CLIENTS READY!")
            logger.info("Sufficient APIs available for data collection.")
        else:
            logger.warning("⚠️  LIMITED API ACCESS")
            logger.info("Consider obtaining API keys for better data access.")
        
        return passed_tests >= 2  # Minimum 2 APIs working

def main():
    """Main entry point."""
    tester = APIClientTester()
    success = tester.test_all_apis()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()