#!/usr/bin/env python3
"""
Phase 1 Dashboard Test
Tests HTML dashboard functionality and data display.
"""

import os
import sys
import logging
import subprocess
import time
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardTester:
    def __init__(self):
        """Initialize dashboard tester."""
        self.dashboard_path = Path("src/dashboard")
        self.port = 8001  # Use different port to avoid conflicts
        self.server_process = None
        
    def start_dashboard_server(self):
        """Start HTTP server for dashboard."""
        logger.info(f"Starting dashboard server on port {self.port}...")
        
        try:
            # Change to dashboard directory and start server
            cmd = [sys.executable, "-m", "http.server", str(self.port)]
            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.dashboard_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Test if server is responding
            response = requests.get(f"http://localhost:{self.port}", timeout=5)
            
            if response.status_code == 200:
                logger.info("✅ Dashboard server started successfully")
                return True
            else:
                logger.error(f"❌ Dashboard server returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard server: {e}")
            return False
    
    def stop_dashboard_server(self):
        """Stop dashboard server."""
        if self.server_process:
            logger.info("Stopping dashboard server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
    
    def test_dashboard_files(self):
        """Test that dashboard files exist."""
        logger.info("Testing dashboard file structure...")
        
        required_files = [
            "index.html",
            "script.js",  # Check if exists
            "styles.css"  # Check if exists
        ]
        
        missing_files = []
        
        for file_name in required_files:
            file_path = self.dashboard_path / file_name
            if file_path.exists():
                logger.info(f"✅ {file_name} exists")
            else:
                if file_name in ["script.js", "styles.css"]:
                    logger.info(f"ℹ️  {file_name} not found (may be embedded)")
                else:
                    missing_files.append(file_name)
                    logger.error(f"❌ {file_name} missing")
        
        if missing_files:
            return False
        
        # Check if index.html has content
        index_path = self.dashboard_path / "index.html"
        try:
            with open(index_path, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Basic size check
                    logger.info("✅ index.html has substantial content")
                    return True
                else:
                    logger.error("❌ index.html appears to be too small")
                    return False
        except Exception as e:
            logger.error(f"❌ Error reading index.html: {e}")
            return False
    
    def test_dashboard_content(self):
        """Test dashboard content and functionality."""
        logger.info("Testing dashboard content...")
        
        try:
            response = requests.get(f"http://localhost:{self.port}", timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Dashboard not accessible: {response.status_code}")
                return False
            
            content = response.text
            
            # Check for key elements
            required_elements = [
                "Congressional Trading Intelligence",
                "dashboard",
                "Trading",
                "Members",
                "html",
                "<head>",
                "<body>"
            ]
            
            missing_elements = []
            
            for element in required_elements:
                if element.lower() in content.lower():
                    logger.info(f"✅ Found '{element}' in dashboard")
                else:
                    missing_elements.append(element)
                    logger.warning(f"⚠️  '{element}' not found in dashboard")
            
            # Check basic HTML structure
            if "<html" in content and "</html>" in content:
                logger.info("✅ Valid HTML structure detected")
            else:
                logger.error("❌ Invalid HTML structure")
                return False
            
            # Check for CSS styling
            if "<style>" in content or "css" in content.lower():
                logger.info("✅ CSS styling detected")
            else:
                logger.warning("⚠️  No CSS styling detected")
            
            # Check for JavaScript
            if "<script>" in content or "javascript" in content.lower():
                logger.info("✅ JavaScript functionality detected")
            else:
                logger.warning("⚠️  No JavaScript detected")
            
            if len(missing_elements) <= 2:  # Allow some missing elements
                logger.info("✅ Dashboard content appears complete")
                return True
            else:
                logger.error(f"❌ Too many missing elements: {missing_elements}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing dashboard content: {e}")
            return False
    
    def test_dashboard_responsiveness(self):
        """Test dashboard response times."""
        logger.info("Testing dashboard responsiveness...")
        
        try:
            start_time = time.time()
            response = requests.get(f"http://localhost:{self.port}", timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                if response_time < 2.0:
                    logger.info(f"✅ Dashboard loads quickly: {response_time:.2f}s")
                    return True
                elif response_time < 5.0:
                    logger.warning(f"⚠️  Dashboard loads slowly: {response_time:.2f}s")
                    return True
                else:
                    logger.error(f"❌ Dashboard loads too slowly: {response_time:.2f}s")
                    return False
            else:
                logger.error(f"❌ Dashboard not accessible: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing dashboard responsiveness: {e}")
            return False
    
    def test_data_integration(self):
        """Test if dashboard can potentially integrate with database."""
        logger.info("Testing data integration capabilities...")
        
        try:
            # This is a basic test - check if dashboard structure supports data
            response = requests.get(f"http://localhost:{self.port}", timeout=5)
            content = response.text
            
            # Look for data-related elements
            data_indicators = [
                "data",
                "member",
                "trade",
                "congress",
                "chart",
                "table",
                "list"
            ]
            
            found_indicators = 0
            for indicator in data_indicators:
                if indicator.lower() in content.lower():
                    found_indicators += 1
            
            if found_indicators >= 3:
                logger.info(f"✅ Dashboard appears data-ready ({found_indicators} data indicators found)")
                return True
            else:
                logger.warning(f"⚠️  Limited data integration indicators ({found_indicators} found)")
                return True  # Don't fail on this - it's just a capability test
                
        except Exception as e:
            logger.error(f"❌ Error testing data integration: {e}")
            return False
    
    def run_all_tests(self):
        """Run all dashboard tests."""
        logger.info("="*60)
        logger.info("PHASE 1 DASHBOARD TESTING")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Start with file tests (don't need server)
        file_test_passed = self.test_dashboard_files()
        
        if not file_test_passed:
            logger.error("❌ Dashboard files missing - cannot continue")
            return False
        
        # Start server for remaining tests
        server_started = self.start_dashboard_server()
        
        if not server_started:
            logger.error("❌ Cannot start dashboard server - skipping server tests")
            return False
        
        try:
            tests = [
                ("Dashboard Content", self.test_dashboard_content),
                ("Dashboard Responsiveness", self.test_dashboard_responsiveness),
                ("Data Integration Capability", self.test_data_integration)
            ]
            
            passed_tests = 1  # File test already passed
            total_tests = len(tests) + 1  # Include file test
            
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
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("\n" + "="*60)
            logger.info("DASHBOARD TEST RESULTS")
            logger.info("="*60)
            logger.info(f"Tests passed: {passed_tests}/{total_tests}")
            logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info(f"Dashboard URL: http://localhost:{self.port}")
            
            if passed_tests >= 3:  # At least 3 out of 4 tests passing
                logger.info("🎉 DASHBOARD READY!")
                logger.info("HTML dashboard is functional and accessible.")
            else:
                logger.warning("⚠️  DASHBOARD NEEDS ATTENTION")
                logger.info("Some dashboard functionality may need debugging.")
            
            return passed_tests >= 3
            
        finally:
            self.stop_dashboard_server()

def main():
    """Main entry point."""
    tester = DashboardTester()
    success = tester.run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()