#!/usr/bin/env python3
"""
Test all example files for basic functionality
"""

import sys
import os
import traceback

def test_python_files():
    """Test Python files can be imported/executed"""
    print("Testing Python files...")
    
    # Test ni.py
    try:
        exec(open('ni.py').read())
        print("✓ ni.py - runs successfully")
    except Exception as e:
        print(f"✗ ni.py - failed: {e}")
    
    # Test watchlist.py
    try:
        exec(open('watchlist.py').read())
        print("✓ watchlist.py - runs successfully")
    except Exception as e:
        print(f"✗ watchlist.py - failed: {e}")

def test_notebook_imports():
    """Test that notebooks can import their dependencies"""
    print("\nTesting notebook import dependencies...")
    
    # Import basic dependencies needed by notebooks
    try:
        import pandas as pd
        import pandas_ta_classic as ta
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        print("✓ Basic dependencies imported successfully")
    except Exception as e:
        print(f"✗ Basic dependencies failed: {e}")
        return False
    
    # Test optional dependencies with error handling
    optional_deps = {
        'tqdm': 'tqdm',
        'mplfinance': 'mplfinance',
    }
    
    for name, import_name in optional_deps.items():
        try:
            __import__(import_name)
            print(f"✓ {name} - available")
        except ImportError:
            print(f"⚠ {name} - not available (install with: pip install {name})")
    
    # Test pandas-ta-classic functionality
    try:
        df = pd.DataFrame({'close': [1,2,3,4,5]})
        result = df.ta.sma(length=2)
        print("✓ pandas-ta-classic - basic functionality works")
    except Exception as e:
        print(f"✗ pandas-ta-classic - failed: {e}")
        return False
    
    # Test watchlist import
    try:
        from watchlist import colors, Watchlist
        test_colors = colors("GrRd")
        print(f"✓ watchlist - import successful, colors work: {test_colors}")
    except Exception as e:
        print(f"✗ watchlist - import failed: {e}")
    
    return True

def test_sample_data_loading():
    """Test loading sample data"""
    print("\nTesting sample data loading...")
    
    try:
        import pandas as pd
        # Load sample data
        data_path = os.path.join('..', 'data', 'SPY_D.csv')
        df = pd.read_csv(data_path, index_col='date', parse_dates=True)
        df.columns = df.columns.str.lower()
        df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')
        
        print(f"✓ Sample data loaded successfully: {df.shape} rows, columns: {list(df.columns)}")
        
        # Test basic TA on sample data
        result = df.ta.sma(length=20)
        print(f"✓ Technical analysis on sample data works: {result.name}")
        
        return True
    except Exception as e:
        print(f"✗ Sample data loading failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing all example files...")
    print("=" * 50)
    
    test_python_files()
    imports_ok = test_notebook_imports()
    data_ok = test_sample_data_loading()
    
    print("\n" + "=" * 50)
    if imports_ok and data_ok:
        print("✓ All core functionality tests passed!")
        print("The examples should work with the available dependencies.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Check the errors above.")
        sys.exit(1)