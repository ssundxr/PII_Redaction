"""
Dataset preparation helper for Privara fine-tuning
"""

import json
import os
from pathlib import Path

def create_trocr_dataset_template():
    """Create TrOCR dataset structure."""
    os.makedirs('datasets/trocr/train_images', exist_ok=True)
    
    # Example labels.json
    labels = {
        "aadhaar_1.jpg": "Shyam Sundar Baskaran",
        "aadhaar_2.jpg": "4366 3802 8529",
        "pan_1.jpg": "ABCDE1234F"
    }
    
    with open('datasets/trocr/labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print("✓ TrOCR dataset template created: datasets/trocr/")
    print("  Add your images to: datasets/trocr/train_images/")
    print("  Edit labels in: datasets/trocr/labels.json")


def create_yolo_dataset_template():
    """Create YOLO dataset structure."""
    for split in ['train', 'val']:
        os.makedirs(f'datasets/yolo/{split}/images', exist_ok=True)
        os.makedirs(f'datasets/yolo/{split}/labels', exist_ok=True)
    
    # Example dataset.yaml
    yaml_content = """
# Privara YOLO Dataset
train: train/images
val: val/images

nc: 4  # number of classes
names: ['face', 'signature', 'qr_code', 'photo']

# Label format (YOLO):
# class_id center_x center_y width height (all normalized 0-1)
# Example: 1 0.5 0.3 0.2 0.1
"""
    
    with open('datasets/yolo/dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✓ YOLO dataset template created: datasets/yolo/")
    print("  Add training images to: datasets/yolo/train/images/")
    print("  Add training labels to: datasets/yolo/train/labels/")
    print("  Label format: class_id center_x center_y width height")


def create_layoutlm_dataset_template():
    """Create LayoutLM dataset structure."""
    os.makedirs('datasets/layoutlm/train_images', exist_ok=True)
    
    # Example annotations.json
    annotations = {
        "aadhaar_1.jpg": {
            "words": ["Name:", "John", "Doe", "Aadhaar:", "1234", "5678", "9012"],
            "boxes": [[100,100,200,120], [250,100,350,120], [380,100,450,120], 
                      [100,150,250,170], [280,150,350,170], [380,150,450,170], [480,150,550,170]],
            "labels": ["O", "B-PERSON", "I-PERSON", "O", "B-AADHAAR", "I-AADHAAR", "I-AADHAAR"]
        }
    }
    
    with open('datasets/layoutlm/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print("✓ LayoutLM dataset template created: datasets/layoutlm/")
    print("  Add your images to: datasets/layoutlm/train_images/")
    print("  Edit annotations in: datasets/layoutlm/annotations.json")


if __name__ == "__main__":
    print("\n📦 Creating dataset templates...\n")
    create_trocr_dataset_template()
    print()
    create_yolo_dataset_template()
    print()
    create_layoutlm_dataset_template()
    print("\n✓ All dataset templates created!")
    print("\nNext steps:")
    print("1. Add your training images to respective folders")
    print("2. Create/edit annotations for each dataset")
    print("3. Run: python train_models.py")
