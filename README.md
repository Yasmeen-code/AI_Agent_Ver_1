# AI Agent for Extracting Names from Images

This project is an AI Agent that reads names from images and saves them in an Excel file using OCR (Optical Character Recognition) technology.

## Features

- ✅ Extract text from images using OCR
- ✅ Support for Arabic and English languages
- ✅ Automatic name extraction from text
- ✅ Export results to Excel file
- ✅ User-friendly interface
- ✅ Image processing to improve reading accuracy

## Requirements

### 1. Python

- Python 3.8 or later

### 2. Tesseract OCR

Tesseract OCR must be installed on the system:

#### Windows:

1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it in the default path
3. Add Arabic language by downloading the `ara.traineddata` file

#### Linux (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ara
```

#### macOS:

```bash
brew install tesseract tesseract-lang
```

## Installation

1. **Clone the project:**

```bash
git clone <repository-url>
cd name-extractor-ai
```

2. **Create a virtual environment:**

```bash
python -m venv venv
```

3. **Activate the virtual environment:**

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

4. **Install requirements:**

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Graphical User Interface

```bash
python h.py
```

### Method 2: Direct code usage

```python
from name_extractor_agent import NameExtractorAgent

# Create the Agent
agent = NameExtractorAgent()

# Process image and save results
result = agent.process_image_to_excel(
    image_path="path/to/your/image.jpg",
    output_file="extracted_names.xlsx"
)

if result['success']:
    print(f"Successfully extracted {len(result['names'])} names!")
    print(f"Saved file: {result['output_file']}")
else:
    print(f"Error occurred: {result['error']}")
```

## Supported Image Types

- JPG/JPEG
- PNG
- BMP
- TIFF

## Usage Example

1. Run the program: `python h.py`
2. Select the image containing names
3. Specify the Excel output file name
4. Click "Extract Names"
5. Wait for processing to complete
6. Find the extracted names in the Excel file

## Excel File Structure

An Excel file will be created containing:

- "No." column: Sequential number for each name
- "Name" column: Extracted names
- "Full Name" column: Copy of names for clarity

## Troubleshooting

### Error: "Tesseract not found"

- Ensure Tesseract OCR is installed
- On Windows, you may need to modify the path in `name_extractor_agent.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Low accuracy in text extraction

- Ensure the image is clear
- Try higher quality images
- Ensure the text is clear and readable

### No names found

- Ensure the image contains clear names
- Try improving image quality
- Ensure names are written clearly

## Contributing

We welcome contributions! You can:

- Report bugs
- Suggest improvements
- Add new features
- Improve name extraction accuracy

## License

This project is licensed under the MIT License.

## Support

If you encounter any issues or have questions, please open an issue in the repository.
# AI_Agent_Ver_1
