"""
test_environment.py
-------------------
Verification script for Day 1: Project Setup.
Validates Python runtime version, required libraries, and directory structure.
"""

import sys
import os
import platform

def check_python_version():
    """Verify that Python version is 3.8 or higher."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    version_str = f"{major}.{minor}.{micro}"
    
    print(f"[*] Python Version detected: {version_str} ({platform.system()} {platform.release()})")
    if major >= 3 and minor >= 8:
        print("    [PASS] Python version is compatible (>= 3.8).\n")
        return True
    else:
        print("    [WARN] Recommended Python version is >= 3.8.\n")
        return False

def check_dependencies():
    """Check availability of required data science and ML packages."""
    required_packages = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "Scikit-Learn",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
    }
    
    print("[*] Checking Required Libraries:")
    all_passed = True
    
    for module_name, display_name in required_packages.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"    [PASS] {display_name:<15} : version {version}")
        except ImportError:
            print(f"    [FAIL] {display_name:<15} : NOT installed (run: pip install -r requirements.txt)")
            all_passed = False
            
    print()
    return all_passed

def check_directory_structure():
    """Verify that expected project directories exist."""
    expected_folders = [
        "data",
        "notebooks",
        "src",
        "models",
        "results",
    ]
    
    print("[*] Checking Project Directory Structure:")
    all_folders_exist = True
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for folder in expected_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            print(f"    [PASS] Directory found: '{folder}/'")
        else:
            print(f"    [FAIL] Directory missing: '{folder}/'")
            all_folders_exist = False
            
    print()
    return all_folders_exist

def main():
    print("=" * 65)
    print("   Sonar Signal Classification System - Environment Check (Day 1)")
    print("=" * 65)
    print()
    
    py_ok = check_python_version()
    dir_ok = check_directory_structure()
    dep_ok = check_dependencies()
    
    print("=" * 65)
    if dir_ok and dep_ok and py_ok:
        print("   STATUS: SUCCESS! All project structures and libraries are ready.")
        print("   Ready to proceed to Day 2.")
    elif dir_ok:
        print("   STATUS: DIRECTORIES READY. Please install missing libraries using:")
        print("           pip install -r requirements.txt")
    else:
        print("   STATUS: PLEASE REVIEW MISSING COMPONENTS ABOVE.")
    print("=" * 65)

if __name__ == "__main__":
    main()
