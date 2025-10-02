"""
Test script to verify name extraction works with loose accuracy
"""
from name_extractor_agent import NameExtractorAgent

def test_with_image():
    agent = NameExtractorAgent()
    result = agent.process_image_to_excel('test_image.jpg', 'test_output.xlsx')
    print("Test Result:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Extracted {len(result['names'])} names:")
        for name in result['names']:
            print(f"  - {name}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_with_image()
