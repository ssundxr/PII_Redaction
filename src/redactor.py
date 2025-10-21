import time
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from difflib import get_close_matches
import numpy as np
from .ocr import TrOCRExtractor
from .detector import VisualPIIDetector
from .privacy.nlp_explain import get_nlp_explainer
from .storage.local_db import get_local_db


def _apply_watermarked_redaction(image: Image.Image, boxes: List[Tuple[int, int, int, int]],
                                 watermark_text: str = "PII REDACTED") -> Image.Image:
    """
    Apply watermarked redaction with padding, solid black fill, and watermarks.

    Args:
        image: PIL Image to redact
        boxes: List of bounding boxes to redact (x, y, w, h format)
        watermark_text: Text to overlay on redacted areas

    Returns:
        PIL Image with watermarked redactions applied
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Process each box individually (no merging)
    for x, y, w, h in boxes:
        # Expand bbox by 5px padding
        new_x = max(0, x - 5)
        new_y = max(0, y - 5)
        new_w = w + 10
        new_h = h + 10

        # Ensure within image bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, img.width - new_x)
        new_h = min(new_h, img.height - new_y)

        # Skip very small boxes
        if new_w < 40 or new_h < 20:
            continue

        # Draw solid black rectangle
        draw.rectangle([new_x, new_y, new_x + new_w, new_y + new_h], fill=(0, 0, 0))

        # Add watermark if box is large enough
        if new_w >= 60 and new_h >= 30:
            try:
                from PIL import ImageFont
                # Try to load a font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except (OSError, IOError):
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", 12)
                    except (OSError, IOError):
                        font = ImageFont.load_default()

                # Calculate text position (centered)
                text_width = draw.textbbox((0, 0), watermark_text, font=font)[2]
                text_height = draw.textbbox((0, 0), watermark_text, font=font)[3]
                text_x = new_x + (new_w - text_width) // 2
                text_y = new_y + (new_h - text_height) // 2

                # Draw white outline for better visibility
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    draw.text((text_x + dx, text_y + dy), watermark_text,
                             fill=(255, 255, 255), font=font)

                # Draw main watermark text
                draw.text((text_x, text_y), watermark_text,
                         fill=(160, 160, 160), font=font)

            except Exception as e:
                # Fallback: simple text without font
                draw.text((new_x + 5, new_y + 5), watermark_text,
                         fill=(160, 160, 160))

    return img


def redact_image(image: Image.Image, boxes: List[Tuple[int, int, int, int]], fill=(0, 0, 0)) -> Image.Image:
    """
    Legacy redaction function - kept for compatibility.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], fill=fill)
    return img


def redact_text(text: str, spans: List[Tuple[int, int]], mask: str = "[*REDACTED*]") -> str:
    if not spans:
        return text
    spans = sorted(spans, key=lambda s: s[0])
    out: List[str] = []
    i = 0
    for start, end in spans:
        out.append(text[i:start])
        out.append(mask)
        i = end
    out.append(text[i:])
    return "".join(out)


class PIIRedactor:
    """
    A comprehensive PII redaction system that combines text and visual PII detection
    to automatically redact sensitive information from document images.

    This class uses TrOCR for text extraction and YOLOv8 for visual PII detection,
    then applies black box redaction to protect sensitive information.
    """

    # PII keywords to filter for in extracted text
    PII_KEYWORDS = [
        "name", "id", "admission", "date of birth", "dob",
        "phone", "email", "address", "ssn", "social security",
        "passport", "license", "account", "card"
    ]

    # PII field labels for key-value pair detection (subset of PII_KEYWORDS)
    PII_LABELS = [
        "admission no", "roll no", "aadhaar", "pan", "account number", "ifsc",
        "mobile", "phone", "email", "address", "dob", "father name", "mother name",
        "spouse name", "nominee", "religion", "caste", "income", "salary", "loan amount"
    ]

    # Class variable for model caching
    _cached_models = {}

    def __init__(self, trocr_model: str = "microsoft/trocr-base-printed",
                 yolo_model: str = "yolov8n.pt",
                 confidence_threshold: float = 0.25):
        """
        Initialize the PIIRedactor with OCR and visual detection models.

        Args:
            trocr_model (str): TrOCR model name or path.
                              Defaults to "microsoft/trocr-base-printed".
            yolo_model (str): YOLOv8 model file path. Defaults to "yolov8n.pt".
            confidence_threshold (float): Confidence threshold for detections.
                                        Defaults to 0.25.

        Raises:
            Exception: If model initialization fails.
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.logger.info("Initializing PIIRedactor...")

            # Initialize text extraction model with caching
            model_key = f"trocr_{trocr_model}"
            if model_key not in PIIRedactor._cached_models:
                self.logger.info(f"Loading TrOCR model: {trocr_model}")
                PIIRedactor._cached_models[model_key] = TrOCRExtractor(model_name=trocr_model)
            self.text_extractor = PIIRedactor._cached_models[model_key]
            self.logger.info("TrOCR extractor initialized")

            # Initialize visual PII detector with caching
            model_key = f"yolo_{yolo_model}"
            if model_key not in PIIRedactor._cached_models:
                self.logger.info(f"Loading YOLO model: {yolo_model}")
                PIIRedactor._cached_models[model_key] = VisualPIIDetector(
                    model_path=yolo_model,
                    confidence_threshold=confidence_threshold
                )
            self.visual_detector = PIIRedactor._cached_models[model_key]
            self.logger.info("Visual PII detector initialized")

            # Initialize LayoutLM detector (optional)
            try:
                from .layoutlm_detector import LayoutLMDetector
                self.layoutlm_detector = LayoutLMDetector()
                self.logger.info("LayoutLM detector initialized")
            except Exception as e:
                self.logger.warning(f"LayoutLM detector not available: {str(e)}")
                self.layoutlm_detector = None

            # Initialize NLP explainer
            self.nlp_explainer = get_nlp_explainer()
            self.logger.info("NLP explainer initialized")

            # Initialize database
            self.db = get_local_db()
            self.logger.info("Local database initialized")

            # Initialize audit log data
            self.audit_data: Dict[str, Any] = {}

        except Exception as e:
            self.logger.error(f"Failed to initialize PIIRedactor: {str(e)}")
            raise Exception(f"PIIRedactor initialization failed: {str(e)}")

    def process_document(self, image_path: str) -> Image.Image:
        """
        Process a document image to detect and redact all PII.

        This method:
        1. Loads the image from the given path
        2. Extracts text lines using TrOCR
        3. Detects visual PII using YOLOv8
        4. Filters text for PII keywords
        5. Combines all bounding boxes
        6. Applies black box redaction

        Args:
            image_path (str): Path to the input image file.

        Returns:
            PIL.Image.Image: The redacted image with PII regions blacked out.

        Raises:
            Exception: If document processing fails.
        """
        try:
            self.logger.info(f"Processing document: {image_path}")

            # Load the image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize image for faster processing (max 2048x2048) - _preprocess_image equivalent
            original_size = image.size
            if max(image.size) > 2048:
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                scale_factor = min(2048 / original_size[0], 2048 / original_size[1])
            else:
                scale_factor = 1.0

            # Initialize audit data for this document
            self.audit_data = {
                "filename": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "num_redactions": 0,
                "pii_items": [],
                "text_pii_count": 0,
                "visual_pii_count": 0,
                "kv_pairs_detected": 0,
                "values_redacted": 0
            }

            text_pii_count = 0
            visual_pii_count = 0
            kv_pairs_count = 0
            values_redacted_count = 0

            # Collect all bounding boxes to redact
            all_boxes = []

            # Extract and analyze text lines
            self.logger.info("Extracting text lines...")
            ocr_start = time.time()
            try:
                text_results = self.text_extractor.extract_text_lines(image)

                # Convert text results to blocks format for key-value detection
                text_blocks = []
                for text, bbox in text_results:
                    text_blocks.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': 0.8  # Default confidence
                    })

                # Extract key-value pairs from structured forms
                kv_pairs = self._extract_key_value_pairs(text_blocks)

                # Add key-value pair detections to redaction boxes and audit
                for kv_pair in kv_pairs:
                    kv_pairs_count += 1

                    # Add value bbox to redaction boxes (redact the actual PII value)
                    if kv_pair['value_bbox']:
                        all_boxes.append(kv_pair['value_bbox'])
                        values_redacted_count += 1

                    # Add key bbox too for complete field redaction
                    all_boxes.append(kv_pair['key_bbox'])

                    # Add to audit log
                    pii_item = {
                        "bbox": kv_pair['key_bbox'],
                        "label": f"KV_{kv_pair['pii_type']}",
                        "reason": f"Key-value pair detection: {kv_pair['label']} -> {kv_pair['value']}",
                        "explanation_text": f"Detected form field '{kv_pair['label']}' with value '{kv_pair['value']}'",
                        "confidence": kv_pair['confidence'],
                        "text_content": f"{kv_pair['label']}: {kv_pair['value']}"
                    }
                    self.audit_data["pii_items"].append(pii_item)

                # Also keep traditional keyword-based detection for non-form text
                text_fragments = []
                bboxes = []

                for text, bbox in text_results:
                    # Skip blocks that are already handled as key-value pairs
                    is_kv_label = any(kv_pair['key_bbox'] == bbox for kv_pair in kv_pairs)
                    is_kv_value = any(kv_pair['value_bbox'] == bbox for kv_pair in kv_pairs)

                    if not is_kv_label and not is_kv_value and self._contains_pii_keywords(text):
                        text_fragments.append(text)
                        bboxes.append(bbox)
                        all_boxes.append(bbox)

                # Batch explain text fragments
                if text_fragments:
                    explanations = self.nlp_explainer.batch_explain(text_fragments, bboxes)

                    for explanation in explanations:
                        pii_item = {
                            "bbox": explanation["bbox"],
                            "label": "TEXT_PII",
                            "reason": "Keyword matching and NLP analysis",
                            "explanation_text": explanation["primary_explanation"],
                            "confidence": 0.85,
                            "text_content": explanation["text"]
                        }
                        self.audit_data["pii_items"].append(pii_item)
                        text_pii_count += 1

                        self.logger.debug(f"PII text detected: '{explanation['text']}' at {explanation['bbox']}")

            except Exception as e:
                self.logger.warning(f"Text extraction failed: {str(e)}")
                # Continue with visual detection even if text extraction fails

            ocr_time = time.time() - ocr_start
            self.logger.info(f"OCR: {ocr_time:.1f}s")

            # Detect visual PII (people in photos)
            self.logger.info("Detecting visual PII...")
            yolo_start = time.time()
            try:
                visual_results = self.visual_detector.detect_visual_pii(image)

                for class_name, bbox, confidence in visual_results:
                    # Convert bbox format from (x1, y1, x2, y2) to (x, y, w, h)
                    x1, y1, x2, y2 = bbox
                    converted_bbox = (x1, y1, x2 - x1, y2 - y1)
                    all_boxes.append(converted_bbox)

                    # Add visual PII item with explanation
                    pii_item = {
                        "bbox": {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1},
                        "label": f"VISUAL_{class_name.upper()}",
                        "reason": "Computer vision detection using YOLOv8",
                        "explanation_text": f"Visual detection of {class_name} in image content",
                        "confidence": float(confidence),
                        "text_content": f"[Visual: {class_name}]"
                    }
                    self.audit_data["pii_items"].append(pii_item)
                    visual_pii_count += 1

                    self.logger.debug(f"Visual PII detected: {class_name} at {bbox} with confidence {confidence:.3f}")

            except Exception as e:
                self.logger.warning(f"Visual PII detection failed: {str(e)}")
                # Continue with redaction of text-based PII if visual detection fails

            yolo_time = time.time() - yolo_start
            self.logger.info(f"YOLO: {yolo_time:.1f}s")

            # Detect structured PII with LayoutLM (if enabled)
            layoutlm_start = time.time()
            if hasattr(self, 'layoutlm_detector') and self.layoutlm_detector:
                try:
                    self.logger.info("Detecting structured PII with LayoutLM...")
                    layoutlm_results = self.layoutlm_detector.detect_structured_pii(image_path)

                    for result in layoutlm_results:
                        # Convert bbox format
                        x, y, w, h = result['bbox']
                        converted_bbox = (x, y, w, h)
                        all_boxes.append(converted_bbox)

                        pii_item = {
                            "bbox": result['bbox'],
                            "label": f"LAYOUTLM_{result['label']}",
                            "reason": "LayoutLMv3 structured document analysis",
                            "explanation_text": f"LayoutLM detected {result['label']} entity",
                            "confidence": result['confidence'],
                            "text_content": result['text']
                        }
                        self.audit_data["pii_items"].append(pii_item)
                        text_pii_count += 1

                        self.logger.debug(f"LayoutLM PII detected: {result['label']} '{result['text']}' at {result['bbox']}")

                except Exception as e:
                    self.logger.warning(f"LayoutLM detection failed: {str(e)}")

            layoutlm_time = time.time() - layoutlm_start
            self.logger.info(f"LayoutLM: {layoutlm_time:.1f}s")

            total_time = ocr_time + yolo_time + layoutlm_time
            self.logger.info(f"Total processing time: {total_time:.1f}s")

            # Apply redaction to all detected PII regions
            self.audit_data["num_redactions"] = len(all_boxes)
            self.audit_data["text_pii_count"] = text_pii_count
            self.audit_data["visual_pii_count"] = visual_pii_count
            self.audit_data["kv_pairs_detected"] = kv_pairs_count
            self.audit_data["values_redacted"] = values_redacted_count
            self.logger.info(f"Applying redaction to {len(all_boxes)} PII regions...")
            redacted_image = _apply_watermarked_redaction(image, all_boxes, "PII REDACTED")

            # Store audit log in encrypted database
            try:
                audit_id = self.db.store_audit(self.audit_data)
                self.logger.info(f"Stored audit log with ID {audit_id}")
            except Exception as e:
                self.logger.warning(f"Failed to store audit log: {e}")

            self.logger.info(f"Document processed successfully. Total redactions: {len(all_boxes)}")

            # Scale back to original size if resized
            if scale_factor != 1.0:
                original_width, original_height = original_size
                redacted_image = redacted_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

            return redacted_image

        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")

    def save_redacted(self, redacted_image: Image.Image, output_path: str) -> str:
        """
        Save the redacted image to a file.

        Args:
            redacted_image (PIL.Image.Image): The redacted image to save.
            output_path (str): Path where the redacted image should be saved.

        Returns:
            str: The path where the image was saved.

        Raises:
            Exception: If saving fails.
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the redacted image
            redacted_image.save(output_path)
            self.logger.info(f"Redacted image saved to: {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save redacted image: {str(e)}")
            raise Exception(f"Failed to save redacted image: {str(e)}")

    def generate_audit_log(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a JSON audit log with details about the redaction process.

        The audit log includes:
        - Original filename
        - Timestamp
        - Number of text PII detected
        - Number of visual PII detected
        - List of detected PII types

        Args:
            output_path (Optional[str]): Path to save the JSON audit log.
                                        If None, returns the dict without saving.

        Returns:
            Dict[str, Any]: The audit log data as a dictionary.

        Raises:
            Exception: If saving the audit log fails.
        """
        try:
            if not self.audit_data:
                self.logger.warning("No audit data available. Process a document first.")
                return {}

            # Add summary statistics
            self.audit_data["total_pii_detected"] = (
                self.audit_data["text_pii_count"] +
                self.audit_data["visual_pii_count"]
            )

            # Save to file if path provided
            if output_path:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(output_path, 'w') as f:
                    json.dump(self.audit_data, f, indent=2)

                self.logger.info(f"Audit log saved to: {output_path}")

            return self.audit_data

        except Exception as e:
            self.logger.error(f"Failed to generate audit log: {str(e)}")
            raise Exception(f"Failed to generate audit log: {str(e)}")

    def _contains_pii_keywords(self, text: str) -> bool:
        """
        Check if text contains any PII keywords.

        Args:
            text (str): Text to check for PII keywords.

        Returns:
            bool: True if PII keywords are found, False otherwise.
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.PII_KEYWORDS)

    def _identify_pii_types(self, text: str) -> List[str]:
        """
        Identify which types of PII are present in the text.

        Args:
            text (str): Text to analyze for PII types.

        Returns:
            List[str]: List of PII types found in the text.
        """
        text_lower = text.lower()
        detected_types = []

        # Map keywords to PII types
        pii_type_map = {
            "name": ["name"],
            "id": ["id", "identification", "admission"],
            "date_of_birth": ["date of birth", "dob", "birth"],
            "phone": ["phone", "mobile", "cell"],
            "email": ["email", "@"],
            "address": ["address", "street", "city", "zip"],
            "ssn": ["ssn", "social security"],
            "passport": ["passport"],
            "license": ["license", "driver"],
            "financial": ["account", "card", "bank"]
        }

        for pii_type, keywords in pii_type_map.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_types.append(pii_type)

        return detected_types

    def _extract_key_value_pairs(self, text_blocks: List[dict]) -> List[dict]:
        """
        Extract key-value pairs from structured form text blocks.

        Args:
            text_blocks: List of OCR text blocks with bounding boxes, sorted by y then x

        Returns:
            List of key-value pairs with labels, values, bounding boxes, and PII types
        """
        kv_pairs = []

        # Sort blocks by y-coordinate (top-to-bottom), then x (left-to-right)
        sorted_blocks = sorted(text_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

        # Map labels to PII types for categorization
        label_to_pii_type = {
            "admission no": "ID_NUMBER", "roll no": "ID_NUMBER", "aadhaar": "ID_NUMBER",
            "pan": "ID_NUMBER", "account number": "FINANCIAL", "ifsc": "FINANCIAL",
            "mobile": "PHONE", "phone": "PHONE", "email": "EMAIL", "address": "ADDRESS",
            "dob": "DATE_OF_BIRTH", "father name": "NAME", "mother name": "NAME",
            "spouse name": "NAME", "nominee": "NAME", "religion": "PERSONAL",
            "caste": "PERSONAL", "income": "FINANCIAL", "salary": "FINANCIAL",
            "loan amount": "FINANCIAL"
        }

        for i, block in enumerate(sorted_blocks):
            text = block['text'].strip()
            bbox = block['bbox']

            # Clean label text (remove punctuation)
            clean_label = text.lower().rstrip(':;').strip()

            # Check if this is a PII label using fuzzy matching
            matches = get_close_matches(clean_label, self.PII_LABELS, n=1, cutoff=0.6)
            if matches:
                label = matches[0]
                pii_type = label_to_pii_type.get(label, "GENERAL_PII")

                # Find value blocks for this label
                value_text = ""
                value_bbox = None

                # Search for right neighbor (next block to the right)
                for j in range(i + 1, min(i + 6, len(sorted_blocks))):  # Limit search
                    next_block = sorted_blocks[j]
                    next_bbox = next_block['bbox']

                    # Check if it's a right neighbor (x > label_x + label_w + 20px gap, within 50px y-overlap)
                    if (next_bbox[0] > bbox[0] + bbox[2] + 20 and
                        abs(next_bbox[1] - bbox[1]) < 50):  # y overlap within 50px
                        value_text = next_block['text'].strip()
                        value_bbox = next_bbox
                        break

                # If no right neighbor found, search below (next blocks below label)
                if not value_text:
                    for j in range(i + 1, min(i + 6, len(sorted_blocks))):
                        next_block = sorted_blocks[j]
                        next_bbox = next_block['bbox']

                        # Check if it's below (y > label_y + label_h, within 100px y-gap, same x-range)
                        if (next_bbox[1] > bbox[1] + bbox[3] and
                            next_bbox[1] - (bbox[1] + bbox[3]) < 100 and
                            abs(next_bbox[0] - bbox[0]) < 200):  # within 200px x-range

                            # Combine multiple lines if they seem to be part of the same value
                            if value_text and next_bbox[1] - (bbox[1] + bbox[3]) < 50:
                                value_text += " " + next_block['text'].strip()
                                # Extend bbox to cover all value lines
                                value_bbox = [
                                    min(value_bbox[0], next_bbox[0]),
                                    min(value_bbox[1], next_bbox[1]),
                                    max(value_bbox[2], next_bbox[2]),
                                    max(value_bbox[3], next_bbox[3])
                                ]
                            else:
                                value_text = next_block['text'].strip()
                                value_bbox = next_bbox

                            # Stop if gap is too large (>150px) or we hit next label
                            if (next_bbox[1] - (bbox[1] + bbox[3]) > 150 or
                                self._is_likely_label(next_block['text'])):
                                break

                # Only include if value is meaningful (>2 chars)
                if value_text and len(value_text) > 2:
                    kv_pairs.append({
                        'label': label,
                        'value': value_text,
                        'key_bbox': bbox,
                        'value_bbox': value_bbox,
                        'pii_type': pii_type,
                        'confidence': block.get('confidence', 0.8)
                    })

        return kv_pairs

    def _is_likely_label(self, text: str) -> bool:
        """Check if text is likely a field label (short, ends with colon, etc.)"""
        text = text.strip().lower()
        return (len(text) < 30 and
                (text.endswith(':') or text.endswith(';') or
                 text in self.PII_LABELS or
                 any(label in text for label in self.PII_LABELS)))

    def process_batch(self, image_paths: List[str], output_dir: str) -> List[str]:
        """
        Process multiple documents in batch.

        Args:
            image_paths (List[str]): List of paths to input images.
            output_dir (str): Directory to save redacted images.

        Returns:
            List[str]: List of paths to saved redacted images.
        """
        processed_files = []

        for image_path in image_paths:
            try:
                # Process the document
                redacted_image = self.process_document(image_path)

                # Generate output filename
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}_redacted.png")

                # Save redacted image
                self.save_redacted(redacted_image, output_path)
                processed_files.append(output_path)

                # Save audit log for this file
                audit_path = os.path.join(output_dir, f"{base_name}_audit.json")
                self.generate_audit_log(audit_path)

            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                continue

        return processed_files
