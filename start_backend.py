#!/usr/bin/env python3
"""
Startup script for the backend server
"""
import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

# Change to the project root directory
os.chdir(os.path.dirname(__file__))

# Import and run the Flask app
from backend.app import app

if __name__ == "__main__":
    print("Starting Forgery Detection Backend Server...")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(host="0.0.0.0", port=5000, debug=True)


