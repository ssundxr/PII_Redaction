"""
Privara Model Fine-Tuning Script
Fine-tune TrOCR, YOLO, and LayoutLM for Indian government documents
"""

import os
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# 1. TROCR FINE-TUNING
# ============================================

def finetune_trocr(
    train_images_dir: str,
    train_labels_file: str,
    output_model_dir: str = "models/trocr-finetuned",
    epochs: int = 10
):
    """
    Fine-tune TrOCR for Indian government documents.
    
    Args:
        train_images_dir: Directory with training images
        train_labels_file: JSON file with {image_name: text} mappings
        output_model_dir: Where to save fine-tuned model
        epochs: Number of training epochs
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
    from datasets import Dataset
    import json
    from PIL import Image
    
    logger.info("="*60)
    logger.info("FINE-TUNING TROCR")
    logger.info("="*60)
    
    # Load base model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    # Load training data
    with open(train_labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Prepare dataset
    train_data = []
    for img_name, text in labels.items():
        img_path = os.path.join(train_images_dir, img_name)
        if os.path.exists(img_path):
            train_data.append({'image': img_path, 'text': text})
    
    logger.info(f"Loaded {len(train_data)} training samples")
    
    def preprocess_function(examples):
        images = [Image.open(img_path).convert('RGB') for img_path in examples['image']]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        
        # Tokenize text
        labels = processor.tokenizer(
            examples['text'],
            padding="max_length",
            max_length=128,
            truncation=True
        ).input_ids
        
        labels = [[label if label != processor.tokenizer.pad_token_id else -100 for label in label_seq] for label_seq in labels]
        
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Create dataset
    dataset = Dataset.from_list(train_data)
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['image', 'text'])
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,
        per_device_train_batch_size=4,
        num_train_epochs=epochs,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # Train
    logger.info("Starting TrOCR training...")
    trainer.train()
    
    # Save
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)
    
    logger.info(f"✓ TrOCR fine-tuned model saved to: {output_model_dir}")


# ============================================
# 2. YOLO FINE-TUNING
# ============================================

def finetune_yolo(
    dataset_yaml: str,
    output_model_path: str = "models/yolo-finetuned.pt",
    epochs: int = 50,
    img_size: int = 640
):
    """
    Fine-tune YOLO for signatures, photos, QR codes in documents.
    
    Args:
        dataset_yaml: Path to dataset.yaml file
        output_model_path: Where to save fine-tuned weights
        epochs: Number of training epochs
        img_size: Image size for training
    
    Dataset structure:
        dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── dataset.yaml
    
    dataset.yaml:
        train: train/images
        val: val/images
        nc: 4  # number of classes
        names: ['face', 'signature', 'qr_code', 'photo']
    """
    from ultralytics import YOLO
    
    logger.info("="*60)
    logger.info("FINE-TUNING YOLO")
    logger.info("="*60)
    
    # Load base YOLO model
    model = YOLO('yolov8n.pt')
    
    logger.info(f"Training YOLO on dataset: {dataset_yaml}")
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        patience=10,
        save=True,
        project='runs/train',
        name='yolo_pii_detector'
    )
    
    # Save best weights
    model.save(output_model_path)
    
    logger.info(f"✓ YOLO fine-tuned model saved to: {output_model_path}")
    
    return model


# ============================================
# 3. LAYOUTLM FINE-TUNING
# ============================================

def finetune_layoutlm(
    train_images_dir: str,
    train_annotations_file: str,
    output_model_dir: str = "models/layoutlm-finetuned",
    epochs: int = 5
):
    """
    Fine-tune LayoutLM for Indian government forms.
    
    Args:
        train_images_dir: Directory with training images
        train_annotations_file: JSON with annotations
        output_model_dir: Where to save fine-tuned model
        epochs: Number of training epochs
    
    Annotation format:
    {
        "image_name.jpg": {
            "words": ["Name:", "John", "Doe", "Aadhaar:", "1234"...],
            "boxes": [[x1,y1,x2,y2], ...],  # Normalized 0-1000
            "labels": ["O", "B-PERSON", "I-PERSON", "O", "B-AADHAAR"...]
        }
    }
    """
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import json
    from PIL import Image
    
    logger.info("="*60)
    logger.info("FINE-TUNING LAYOUTLM")
    logger.info("="*60)
    
    # Load base model
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=13)
    
    # Label mapping
    label2id = {
        "O": 0, "B-PERSON": 1, "I-PERSON": 2,
        "B-AADHAAR": 3, "I-AADHAAR": 4,
        "B-PHONE": 5, "I-PHONE": 6,
        "B-EMAIL": 7, "I-EMAIL": 8,
        "B-ADDRESS": 9, "I-ADDRESS": 10,
        "B-DATE": 11, "I-DATE": 12
    }
    
    # Load annotations
    with open(train_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Prepare dataset
    train_data = []
    for img_name, annot in annotations.items():
        img_path = os.path.join(train_images_dir, img_name)
        if os.path.exists(img_path):
            train_data.append({
                'image': img_path,
                'words': annot['words'],
                'boxes': annot['boxes'],
                'labels': [label2id[l] for l in annot['labels']]
            })
    
    logger.info(f"Loaded {len(train_data)} training samples")
    
    def preprocess_function(examples):
        images = [Image.open(img_path).convert('RGB') for img_path in examples['image']]
        
        encoding = processor(
            images,
            examples['words'],
            boxes=examples['boxes'],
            word_labels=examples['labels'],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        
        return encoding
    
    # Create dataset
    dataset = Dataset.from_list(train_data)
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['image', 'words', 'boxes', 'labels'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        save_steps=200,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # Train
    logger.info("Starting LayoutLM training...")
    trainer.train()
    
    # Save
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)
    
    logger.info(f"✓ LayoutLM fine-tuned model saved to: {output_model_dir}")


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def main():
    """Run full fine-tuning pipeline."""
    
    print("\n" + "="*60)
    print("PRIVARA MODEL FINE-TUNING PIPELINE")
    print("="*60 + "\n")
    
    # Configuration
    config = {
        # TrOCR
        'trocr_train_images': 'datasets/trocr/train_images',
        'trocr_labels': 'datasets/trocr/labels.json',
        'trocr_output': 'models/trocr-indian-docs',
        'trocr_epochs': 10,
        
        # YOLO
        'yolo_dataset_yaml': 'datasets/yolo/dataset.yaml',
        'yolo_output': 'models/yolo-pii-detector.pt',
        'yolo_epochs': 50,
        
        # LayoutLM
        'layoutlm_train_images': 'datasets/layoutlm/train_images',
        'layoutlm_annotations': 'datasets/layoutlm/annotations.json',
        'layoutlm_output': 'models/layoutlm-indian-forms',
        'layoutlm_epochs': 5
    }
    
    # 1. Fine-tune TrOCR
    print("\n1/3: Fine-tuning TrOCR...")
    if os.path.exists(config['trocr_train_images']):
        finetune_trocr(
            train_images_dir=config['trocr_train_images'],
            train_labels_file=config['trocr_labels'],
            output_model_dir=config['trocr_output'],
            epochs=config['trocr_epochs']
        )
    else:
        print("⚠ TrOCR dataset not found, skipping...")
    
    # 2. Fine-tune YOLO
    print("\n2/3: Fine-tuning YOLO...")
    if os.path.exists(config['yolo_dataset_yaml']):
        finetune_yolo(
            dataset_yaml=config['yolo_dataset_yaml'],
            output_model_path=config['yolo_output'],
            epochs=config['yolo_epochs']
        )
    else:
        print("⚠ YOLO dataset not found, skipping...")
    
    # 3. Fine-tune LayoutLM
    print("\n3/3: Fine-tuning LayoutLM...")
    if os.path.exists(config['layoutlm_train_images']):
        finetune_layoutlm(
            train_images_dir=config['layoutlm_train_images'],
            train_annotations_file=config['layoutlm_annotations'],
            output_model_dir=config['layoutlm_output'],
            epochs=config['layoutlm_epochs']
        )
    else:
        print("⚠ LayoutLM dataset not found, skipping...")
    
    print("\n" + "="*60)
    print("✓ FINE-TUNING COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
