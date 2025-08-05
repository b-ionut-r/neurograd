#!/usr/bin/env python3
"""
Build script for NeuroGrad package

This script helps build, test, and upload the NeuroGrad package.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result

def clean():
    """Clean build artifacts"""
    print("Cleaning build artifacts...")
    
    # Remove build directories
    dirs_to_remove = [
        "build",
        "dist", 
        "neurograd.egg-info",
        "__pycache__",
        ".pytest_cache",
        "htmlcov",
        ".coverage",
    ]
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}")
    
    # Remove __pycache__ directories recursively
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                full_path = os.path.join(root, dir_name)
                shutil.rmtree(full_path)
                print(f"  Removed {full_path}")
    
    print("✅ Cleanup complete")

def test():
    """Run tests"""
    print("Running tests...")
    
    # Run the installation test
    print("Running installation test...")
    result = run_command("python test_installation.py", check=False)
    if result.returncode != 0:
        print("❌ Installation test failed")
        return False
    
    # Run pytest if available
    try:
        run_command("python -m pytest", check=False)
    except:
        print("pytest not available, skipping unit tests")
    
    print("✅ Tests complete")
    return True

def build():
    """Build the package"""
    print("Building package...")
    
    # Clean first
    clean()
    
    # Build source distribution and wheel
    run_command("python -m build")
    
    # Check the built package
    print("Checking built package...")
    run_command("python -m twine check dist/*")
    
    print("✅ Build complete")
    
    # List built files
    print("\nBuilt files:")
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            print(f"  dist/{file}")

def install_dev():
    """Install in development mode"""
    print("Installing in development mode...")
    run_command("pip install -e .[all]")
    print("✅ Development installation complete")

def upload_test():
    """Upload to test PyPI"""
    print("Uploading to test PyPI...")
    print("Make sure you have set up your ~/.pypirc file with test PyPI credentials")
    run_command("python -m twine upload --repository testpypi dist/*")
    print("✅ Upload to test PyPI complete")

def upload():
    """Upload to PyPI"""
    print("Uploading to PyPI...")
    print("⚠️  This will upload to the REAL PyPI. Make sure you're ready!")
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Upload cancelled")
        return
    
    run_command("python -m twine upload dist/*")
    print("✅ Upload to PyPI complete")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python build.py <command>")
        print("Commands:")
        print("  clean      - Clean build artifacts")
        print("  test       - Run tests")
        print("  build      - Build package")
        print("  install    - Install in development mode")
        print("  upload-test- Upload to test PyPI")
        print("  upload     - Upload to PyPI")
        print("  all        - Clean, test, and build")
        return
    
    command = sys.argv[1].lower()
    
    if command == "clean":
        clean()
    elif command == "test":
        test()
    elif command == "build":
        build()
    elif command == "install":
        install_dev()
    elif command == "upload-test":
        upload_test()
    elif command == "upload":
        upload()
    elif command == "all":
        clean()
        if test():
            build()
        else:
            print("❌ Tests failed, skipping build")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
