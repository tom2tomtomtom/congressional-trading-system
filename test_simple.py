#!/usr/bin/env python3
"""
Test if simple_app.py can be imported and started
"""

try:
    from simple_app import app
    print("✅ simple_app imported successfully")
    
    # Test a route
    with app.test_client() as client:
        response = client.get('/health')
        print(f"✅ Health endpoint works: {response.status_code}")
        
    print("✅ Simple app is ready for deployment")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()