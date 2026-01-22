#!/usr/bin/env python3
"""
Startup script for the complete forgery detection system
"""
import subprocess
import time
import sys
import os

def start_backend():
    """Start the backend server"""
    print("🚀 Starting Backend Server...")
    try:
        # Start backend in background
        subprocess.Popen([sys.executable, "working_backend.py"], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
        print("✅ Backend server started successfully!")
        print("   Backend URL: http://localhost:5000")
        return True
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def start_frontend():
    """Start the frontend server"""
    print("\n🚀 Starting Frontend Server...")
    try:
        # Change to frontend directory and start React app
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        subprocess.Popen(["npm", "start"], 
                        cwd=frontend_dir,
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
        print("✅ Frontend server started successfully!")
        print("   Frontend URL: http://localhost:3000")
        return True
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return False

def main():
    print("=" * 50)
    print("🔍 FORGERY DETECTION SYSTEM STARTUP")
    print("=" * 50)
    
    # Start backend
    backend_success = start_backend()
    
    if backend_success:
        print("\n⏳ Waiting for backend to initialize...")
        time.sleep(3)
        
        # Start frontend
        frontend_success = start_frontend()
        
        if frontend_success:
            print("\n" + "=" * 50)
            print("🎉 SYSTEM STARTED SUCCESSFULLY!")
            print("=" * 50)
            print("📱 Frontend: http://localhost:3000")
            print("🔧 Backend:  http://localhost:5000")
            print("\n💡 Instructions:")
            print("   1. Open http://localhost:3000 in your browser")
            print("   2. Navigate to the upload page")
            print("   3. Upload an image to test forgery detection")
            print("   4. View results and history")
            print("\n⚠️  Press Ctrl+C to stop all servers")
        else:
            print("\n❌ Frontend failed to start. Please run 'npm start' manually in the frontend directory.")
    else:
        print("\n❌ Backend failed to start. Please check the error messages above.")

if __name__ == "__main__":
    main()


