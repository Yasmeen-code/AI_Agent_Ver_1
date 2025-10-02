"""
Simple script to run the Name Extractor AI Agent
"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from h import main
    
    if __name__ == "__main__":
        print("🚀 Starting AI Agent for Name Extraction from Images...")
        print("📋 Make sure all requirements are installed:")
        print("   - pip install -r requirements.txt")
        print("   - Install Tesseract OCR")
        print("   - Download Arabic language for Tesseract")
        print("-" * 50)
        
        main()
        
except ImportError as e:
    print(f"❌ Error importing libraries: {e}")
    print("📦 Make sure requirements are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ General error: {e}")
    sys.exit(1)
