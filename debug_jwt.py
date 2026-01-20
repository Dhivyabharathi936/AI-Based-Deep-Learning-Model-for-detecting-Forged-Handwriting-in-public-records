#!/usr/bin/env python3
"""
Debug JWT token issues
"""
import requests
import json

def debug_jwt():
    base_url = "http://127.0.0.1:5000"
    
    print("=== JWT Debug Test ===")
    
    # Step 1: Login
    print("\n1. Logging in...")
    login_data = {"username": "testuser", "password": "testpass123"}
    
    try:
        response = requests.post(f"{base_url}/login", json=login_data)
        print(f"Login status: {response.status_code}")
        print(f"Login response: {response.json()}")
        
        if response.status_code == 200:
            token = response.json().get("token")
            print(f"Token received: {token[:50]}...")
            
            # Step 2: Test token with simple request
            print("\n2. Testing token with health check...")
            headers = {'Authorization': f'Bearer {token}'}
            
            # Test with a simple GET request first
            response = requests.get(f"{base_url}/", headers=headers)
            print(f"Health check with token: {response.status_code}")
            print(f"Response: {response.json()}")
            
            # Test history endpoint
            print("\n3. Testing history endpoint...")
            response = requests.get(f"{base_url}/history", headers=headers)
            print(f"History status: {response.status_code}")
            if response.status_code != 200:
                print(f"History error: {response.json()}")
            else:
                print(f"History success: {len(response.json().get('history', []))} items")
                
        else:
            print("Login failed!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_jwt()

