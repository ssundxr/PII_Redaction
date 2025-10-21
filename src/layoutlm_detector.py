from typing import List, Dict, Any
import pytesseract
import logging
from PIL import Image
import torch
import numpy as np

try:
    from transformers import LayoutLMv3ImageProcessor, LayoutLMv3ForTokenClassification
    from datasets import Features, Sequence, ClassLabel
    LAYOUTLM_AVAILABLE = True
except ImportError:
    LAYOUTLM_AVAILABLE = False

class LayoutLMDetector:
    """
    A structured document PII detection class that uses LayoutLMv3 to detect
    entities like PERSON, ORG, DATE, ID, etc. in banking forms and documents.
    """

    # LayoutLM entity labels for PII detection
    PII_ENTITIES = {
        "PERSON", "ORG", "GPE", "DATE", "MONEY", "CARDINAL",
        "ID", "PHONE", "EMAIL", "ADDRESS"
    }

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base-finetuned-funsd"):
        """
        Initialize the LayoutLMDetector with the specified model.

        Args:
            model_name (str): LayoutLMv3 model name or path.

        Raises:
            Exception: If LayoutLM is not available or model fails to load.
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name

        if not LAYOUTLM_AVAILABLE:
            raise Exception("LayoutLMv3 dependencies not available. Install with: pip install layoutlmv3 transformers[torch] datasets")

        try:
            self.logger.info(f"Loading LayoutLM model: {model_name}")
            self.processor = LayoutLMv3ImageProcessor.from_pretrained(model_name)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

            # Set device (GPU if available, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            if self.device.type == "cuda":
                self.model.half()  # Use half precision for speed

            self.logger.info(f"LayoutLM model loaded successfully on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load LayoutLM model {model_name}: {str(e)}")
            raise Exception(f"Failed to initialize LayoutLMDetector: {str(e)}")

    def detect_structured_pii(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect structured PII in a document using LayoutLMv3.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            List[Dict[str, Any]]: List of detected PII entities with bbox, label, confidence, text.

        Raises:
            Exception: If detection fails.
        """
        try:
            self.logger.info("Starting LayoutLM structured PII detection")

            # Load image
            image = Image.open(image_path).convert('RGB')

            # Use Tesseract to get words with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Extract words and bounding boxes
            words = []
            bboxes = []

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 60:  # Minimum confidence
                    x = int(data['left'][i])
                    y = int(data['top'][i])
                    w = int(data['width'][i])
                    h = int(data['height'][i])
                    words.append(text)
                    bboxes.append([x, y, x + w, y + h])

            if not words:
                return []

            # Prepare inputs for LayoutLM
            encoding = self.processor(
                image,
                words=words,
                boxes=bboxes,
                return_tensors="pt"
            )

            # Move to device
            for k, v in encoding.items():
                encoding[k] = v.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoding)

            # Get predictions
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = encoding.bbox.squeeze().tolist()

            # Process results
            results = []
            current_word = ""
            current_bbox = None
            current_label = None

            for token_idx, (prediction, bbox) in enumerate(zip(predictions, token_boxes)):
                if token_idx >= len(words):
                    break

                word = words[token_idx]
                label_id = prediction

                # Get label name (assuming standard LayoutLM labels)
                label_map = self.model.config.id2label
                label = label_map.get(label_id, "O")

                if label in self.PII_ENTITIES:
                    if current_word and current_bbox:
                        # Save previous entity
                        results.append({
                            'text': current_word.strip(),
                            'bbox': current_bbox,
                            'label': current_label,
                            'confidence': 0.8  # LayoutLM confidence
                        })

                    # Start new entity
                    current_word = word
                    current_bbox = bbox
                    current_label = label
                elif current_word:
                    # Continue current entity
                    current_word += " " + word
                    # Update bbox to encompass the word
                    current_bbox = [
                        min(current_bbox[0], bbox[0]),
                        min(current_bbox[1], bbox[1]),
                        max(current_bbox[2], bbox[2]),
                        max(current_bbox[3], bbox[3])
                    ]
                else:
                    # No current entity, save standalone word if it's PII
                    if label in self.PII_ENTITIES:
                        results.append({
                            'text': word,
                            'bbox': bbox,
                            'label': label,
                            'confidence': 0.7
                        })

            # Add final entity if exists
            if current_word and current_bbox:
                results.append({
                    'text': current_word.strip(),
                    'bbox': current_bbox,
                    'label': current_label,
                    'confidence': 0.8
                })

            # Filter by confidence threshold and size
            filtered_results = []
            for result in results:
                if result['confidence'] >= 0.7 and result['bbox'][2] - result['bbox'][0] >= 10:
                    filtered_results.append(result)

            self.logger.info(f"LayoutLM detected {len(filtered_results)} structured PII entities")
            return filtered_results

        except Exception as e:
            self.logger.error(f"LayoutLM detection failed: {str(e)}")
            raise Exception(f"Failed to detect structured PII: {str(e)}")
