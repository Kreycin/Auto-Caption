#!/usr/bin/env python3
"""
Build script for AI Auto-Caption Desktop App (Mac)
Creates a standalone .app bundle using PyInstaller
"""

import subprocess
import sys
import os

def main():
    print("=" * 50)
    print("AI Auto-Caption - Mac App Builder")
    print("=" * 50)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("✅ PyInstaller found")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    webapp_path = os.path.join(script_dir, "webapp.py")
    config_path = os.path.join(script_dir, "config.json")
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=AI Auto-Caption",
        "--onedir",  # Create a directory bundle
        "--windowed",  # No console window (GUI app)
        "--noconfirm",  # Overwrite without asking
        f"--add-data={config_path}:.",  # Include config.json
        "--hidden-import=flask",
        "--hidden-import=requests",
        "--hidden-import=pythainlp",
        "--hidden-import=pythainlp.tokenize",
        "--hidden-import=pythainlp.corpus",
        "--hidden-import=pythainlp.util",
        "--collect-data=pythainlp",
        "--collect-submodules=pythainlp",
        webapp_path
    ]
    
    print("\n🔨 Building Mac App...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.check_call(cmd, cwd=script_dir)
        print("\n" + "=" * 50)
        print("✅ Build successful!")
        print(f"📁 App location: {script_dir}/dist/AI Auto-Caption.app")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
