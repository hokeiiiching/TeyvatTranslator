#!/usr/bin/env python3
"""
Build script for TeyvatTranslator
Creates a standalone Windows executable using PyInstaller

Usage:
    python build.py           # Build the executable
    python build.py --clean   # Clean build artifacts first
    python build.py --zip     # Also create a distributable ZIP
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# Build configuration
APP_NAME = "TeyvatTranslator"
VERSION = "1.0.0"
BUILD_DIR = Path("build")
DIST_DIR = Path("dist")
SPEC_FILE = Path("TeyvatTranslator.spec")


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nFailed: {description}")
        return False
    
    print(f"\nCompleted: {description}")
    return True


def clean_build():
    """Remove previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    
    for path in [BUILD_DIR, DIST_DIR]:
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed: {path}")
    
    print("Clean complete")


def install_build_dependencies():
    """Install PyInstaller if not present."""
    try:
        import PyInstaller
        print("PyInstaller is installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)


def build_executable():
    """Build the executable using PyInstaller."""
    if not SPEC_FILE.exists():
        print(f"Spec file not found: {SPEC_FILE}")
        return False
    
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"]
    return run_command(cmd, "Building executable with PyInstaller")


def create_zip():
    """Create a distributable ZIP file."""
    dist_folder = DIST_DIR / APP_NAME
    if not dist_folder.exists():
        print(f"Distribution folder not found: {dist_folder}")
        return False
    
    zip_name = f"{APP_NAME}-v{VERSION}-windows"
    zip_path = DIST_DIR / zip_name
    
    print(f"\nCreating ZIP: {zip_name}.zip")
    shutil.make_archive(str(zip_path), 'zip', DIST_DIR, APP_NAME)
    
    final_zip = Path(f"{zip_path}.zip")
    size_mb = final_zip.stat().st_size / (1024 * 1024)
    print(f"Created: {final_zip} ({size_mb:.1f} MB)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TeyvatTranslator executable")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    parser.add_argument("--zip", action="store_true", help="Create distributable ZIP")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           TeyvatTranslator Build Script                  ║
║                   Version {VERSION}                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Clean if requested
    if args.clean:
        clean_build()
    
    # Install dependencies
    install_build_dependencies()
    
    # Build
    if not build_executable():
        print("\nBuild failed!")
        sys.exit(1)
    
    # Create ZIP if requested
    if args.zip:
        if not create_zip():
            print("\nZIP creation failed, but executable was built")
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    BUILD COMPLETE!                       ║
╠══════════════════════════════════════════════════════════╣
║  Executable: dist/{APP_NAME}/{APP_NAME}.exe       ║
║                                                          ║
║  To distribute:                                          ║
║  1. Run: python build.py --zip                           ║
║  2. Upload the ZIP to GitHub Releases                    ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
