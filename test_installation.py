#!/usr/bin/env python3
"""
Quick test script to verify installation and configuration.
Run this after installing requirements to ensure everything is set up correctly.
"""

import sys
import importlib.util

def test_import(module_name):
    """Test if a module can be imported"""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, f"❌ {module_name} not installed"
    else:
        return True, f"✅ {module_name} installed"

def main():
    print("=" * 50)
    print("Profit Drivers Dashboard - Installation Test")
    print("=" * 50)
    print()
    
    # Test required modules
    required_modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'openpyxl'
    ]
    
    all_good = True
    print("Testing required packages:")
    for module in required_modules:
        success, message = test_import(module)
        print(f"  {message}")
        if not success:
            all_good = False
    
    print()
    
    # Test local modules
    print("Testing local modules:")
    try:
        import data_processor
        print("  ✅ data_processor.py loads correctly")
    except ImportError as e:
        print(f"  ❌ data_processor.py failed to load: {e}")
        all_good = False
    
    print()
    
    # Check for required files
    print("Checking required files:")
    import os
    
    required_files = [
        'data.csv',
        'sector_map.pkl',
        '.streamlit/config.toml'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} exists")
        else:
            print(f"  ❌ {file_path} not found")
            all_good = False
    
    print()
    print("=" * 50)
    
    if all_good:
        print("✅ All tests passed! You can run the dashboard with:")
        print("   streamlit run profit_drivers_dashboard.py")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    print("=" * 50)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())