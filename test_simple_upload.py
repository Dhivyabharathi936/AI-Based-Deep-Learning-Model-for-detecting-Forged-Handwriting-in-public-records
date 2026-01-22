#!/usr/bin/env python3
"""
Test simple upload functionality without JWT
"""
import requests
import json
import os
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (224, 224), color='white')
    # Add some simple content to make it look like a signature
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.line([(50, 100), (150, 100), (100, 150)], fill='black', width=3)
    draw.text((80, 180), "Test", fill='black')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def test_simple_upload():
    base_url = "http://127.0.0.1:5000"
    
    print("=== Simple Upload Test ===")
    
    # Step 1: Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Step 2: Test upload with image
    print("\n2. Testing image upload...")
    
    # Create test image
    test_image = create_test_image()
    
    files = {
        'image': ('test_signature.png', test_image, 'image/png')
    }
    
    try:
        response = requests.post(f"{base_url}/predict", files=files)
        print(f"Upload response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Prediction: {result.get('prediction')}")
            print(f"   Confidence: {result.get('confidence'):.3f}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
    except Exception as e:
        print(f"Upload error: {e}")
        return False

def test_history():
    base_url = "http://127.0.0.1:5000"
    
    print("\n3. Testing history endpoint...")
    
    try:
        response = requests.get(f"{base_url}/history")
        if response.status_code == 200:
            history = response.json().get("history", [])
            print(f"✅ History retrieved: {len(history)} items")
            for item in history[:3]:  # Show first 3 items
                print(f"   - {item.get('filename')}: {item.get('prediction')} ({item.get('confidence'):.3f})")
            return True
        else:
            print(f"❌ History failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
    except Exception as e:
        print(f"History error: {e}")
        return False

if __name__ == "__main__":
    print("=== Simple Forgery Detection Upload Test ===")
    
    # Test upload
    upload_success = test_simple_upload()
    
    # Test history
    history_success = test_history()
    
    print(f"\n=== Test Results ===")
    print(f"Upload: {'✅ PASS' if upload_success else '❌ FAIL'}")
    print(f"History: {'✅ PASS' if history_success else '❌ FAIL'}")
    
    if upload_success and history_success:
        print("\n🎉 All tests passed! The upload functionality is working correctly.")
        print("You can now use the frontend to upload images.")
    else:
        print("\n⚠️  Some tests failed. Please check the backend logs for details.")

