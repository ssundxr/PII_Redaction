import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from detector import VisualPIIDetector
from PIL import Image

# Initialize the visual PII detector
print("Initializing VisualPIIDetector...")
detector = VisualPIIDetector()

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
        results = detector.detect_visual_pii(img)
        
        print(f"\n=== {img_file} ===")
        if results:
            for class_name, bbox, confidence in results:
                print(f"Detected: {class_name}")
                print(f"Bbox: {bbox}")
                print(f"Confidence: {confidence:.3f}")
        else:
            print("No visual PII detected")
            
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
