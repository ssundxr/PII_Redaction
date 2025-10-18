import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from ocr import TrOCRExtractor
from PIL import Image

extractor = TrOCRExtractor()

# Test on all sample images
sample_dir = "tests/sample_images/"
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}

for img_file in os.listdir(sample_dir):
    # Skip non-image files like .gitkeep
    if not any(img_file.lower().endswith(ext) for ext in image_extensions):
        continue
        
    img_path = os.path.join(sample_dir, img_file)
    try:
        img = Image.open(img_path)
        results = extractor.extract_text_lines(img)
        
        print(f"\n=== {img_file} ===")
        for text, bbox in results:
            print(f"Text: {text}")
            print(f"Bbox: {bbox}")
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
