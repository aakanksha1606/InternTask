#!/usr/bin/env python3
"""
Launch script for the Sentiment-Based Text Generator
This script provides an easy way to run the application with proper configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import transformers
        import torch
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def run_application():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting Sentiment-Based Text Generator...")
        print("📱 The application will open in your default browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Run streamlit with optimized settings
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--server.headless=false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install: pip install streamlit")
        return False
    
    return True

def main():
    """Main function to launch the application"""
    print("🤖 Sentiment-Based Text Generator Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory")
        print("Please make sure you're in the correct directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 To install requirements, run:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the application
    if run_application():
        print("✅ Application completed successfully")
    else:
        print("❌ Application failed to start")
        sys.exit(1)

if __name__ == "__main__":
    main()
