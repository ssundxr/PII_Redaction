import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.redactor import PIIRedactor
from PIL import Image

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test_pii_redactor():
    """Test the PIIRedactor class with sample images."""
    
    print("=== PIIRedactor Test ===")
    
    try:
        # Initialize the PIIRedactor
        print("Initializing PIIRedactor...")
        redactor = PIIRedactor()
        print("PIIRedactor initialized successfully!")
        
        # Test on sample images
        sample_dir = "tests/sample_images/"
        output_dir = "tests/output/"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
        image_files = []
        
        for img_file in os.listdir(sample_dir):
            if any(img_file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(sample_dir, img_file))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in image_files:
            try:
                print(f"\n--- Processing: {os.path.basename(img_path)} ---")
                
                # Process the document
                redacted_image = redactor.process_document(img_path)
                
                # Generate output paths
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_redacted.png")
                audit_path = os.path.join(output_dir, f"{base_name}_audit.json")
                
                # Save redacted image
                redactor.save_redacted(redacted_image, output_path)
                
                # Generate and save audit log
                audit_data = redactor.generate_audit_log(audit_path)
                
                # Print summary
                print(f"   Processed successfully!")
                print(f"   Text PII detected: {audit_data.get('text_pii_count', 0)}")
                print(f"   Visual PII detected: {audit_data.get('visual_pii_count', 0)}")
                print(f"   PII types found: {audit_data.get('detected_pii_types', [])}")
                print(f"   Redacted image saved: {output_path}")
                print(f"   Audit log saved: {audit_path}")
                
            except Exception as e:
                print(f" Error processing {img_path}: {e}")
                continue
        
        print(f"\n=== Test completed! Check the '{output_dir}' folder for results ===")
        
    except Exception as e:
        print(f"Failed to initialize PIIRedactor: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_pii_redactor()
