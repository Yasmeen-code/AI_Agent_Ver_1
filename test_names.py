"""
Test script to debug name extraction
"""
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from name_extractor_agent import NameExtractorAgent

def test_name_extraction():
    """Test name extraction with sample data"""
    print("🧪 Testing name extraction...")
    
    # Initialize agent
    agent = NameExtractorAgent()
    
    # Test with sample text from the image description
    sample_text = """
Student Name    YOO
Besarick, Kayla Kathryn
Day, Jenna Aurelia
Doehling, Lauren Christine
Gaynor, Emily Grace Suo
Karen Forti, Brian
Leary, Colin Dana
Leavitt, Mia Jane
Tavares, Dennis
Wisgirda, Peter Michael
Wooten, Sandy John
"""
    
    print("📝 Sample text:")
    print(sample_text)
    print("-" * 50)
    
    # Test extraction
    result = agent.test_name_extraction(sample_text)
    
    print(f"📊 Results:")
    print(f"Number of extracted names: {result['count']}")
    print(f"Names: {result['extracted_names']}")
    
    if result['count'] > 0:
        print("✅ Name extraction successful!")
    else:
        print("❌ Name extraction failed")
        
        # Let's test individual patterns
        print("\n🔍 Testing individual patterns...")
        
        import re
        
        # Test each pattern
        for i, pattern in enumerate(agent.name_patterns['english']):
            matches = re.findall(pattern, sample_text)
            print(f"Pattern {i+1}: {pattern}")
            print(f"  Matches: {matches}")
            print()

if __name__ == "__main__":
    test_name_extraction()
