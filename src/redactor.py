from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from .ocr import TrOCRExtractor
from .detector import VisualPIIDetector


def redact_image(image: Image.Image, boxes: List[Tuple[int, int, int, int]], fill=(0, 0, 0)) -> Image.Image:
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
    
    def __init__(self, trocr_model: str = "microsoft/trocr-base-printed", 
                 yolo_model: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5):
        """
        Initialize the PIIRedactor with OCR and visual detection models.
        
        Args:
            trocr_model (str): TrOCR model name or path. 
                              Defaults to "microsoft/trocr-base-printed".
            yolo_model (str): YOLOv8 model file path. Defaults to "yolov8n.pt".
            confidence_threshold (float): Confidence threshold for detections.
                                        Defaults to 0.5.
        
        Raises:
            Exception: If model initialization fails.
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info("Initializing PIIRedactor...")
            
            # Initialize text extraction model
            self.text_extractor = TrOCRExtractor(model_name=trocr_model)
            self.logger.info("TrOCR extractor initialized")
            
            # Initialize visual PII detector
            self.visual_detector = VisualPIIDetector(
                model_path=yolo_model, 
                confidence_threshold=confidence_threshold
            )
            self.logger.info("Visual PII detector initialized")
            
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
            
            # Initialize audit data for this document
            self.audit_data = {
                "original_filename": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "text_pii_count": 0,
                "visual_pii_count": 0,
                "detected_pii_types": []
            }
            
            # Collect all bounding boxes to redact
            all_boxes = []
            
            # Extract and analyze text lines
            self.logger.info("Extracting text lines...")
            try:
                text_results = self.text_extractor.extract_text_lines(image)
                
                # Filter text lines for PII keywords
                for text, bbox in text_results:
                    if self._contains_pii_keywords(text):
                        all_boxes.append(bbox)
                        self.audit_data["text_pii_count"] += 1
                        
                        # Identify which PII types were detected
                        detected_types = self._identify_pii_types(text)
                        for pii_type in detected_types:
                            if pii_type not in self.audit_data["detected_pii_types"]:
                                self.audit_data["detected_pii_types"].append(pii_type)
                        
                        self.logger.debug(f"PII text detected: '{text}' at {bbox}")
                
            except Exception as e:
                self.logger.warning(f"Text extraction failed: {str(e)}")
                # Continue with visual detection even if text extraction fails
            
            # Detect visual PII (people in photos)
            self.logger.info("Detecting visual PII...")
            try:
                visual_results = self.visual_detector.detect_visual_pii(image)
                
                for class_name, bbox, confidence in visual_results:
                    # Convert bbox format from (x1, y1, x2, y2) to (x, y, w, h)
                    x1, y1, x2, y2 = bbox
                    converted_bbox = (x1, y1, x2 - x1, y2 - y1)
                    all_boxes.append(converted_bbox)
                    self.audit_data["visual_pii_count"] += 1
                    
                    if "person/photo" not in self.audit_data["detected_pii_types"]:
                        self.audit_data["detected_pii_types"].append("person/photo")
                    
                    self.logger.debug(f"Visual PII detected: {class_name} at {bbox} with confidence {confidence:.3f}")
                    
            except Exception as e:
                self.logger.warning(f"Visual PII detection failed: {str(e)}")
                # Continue with redaction of text-based PII if visual detection fails
            
            # Apply redaction to all detected PII regions
            self.logger.info(f"Applying redaction to {len(all_boxes)} PII regions...")
            redacted_image = redact_image(image, all_boxes, fill=(0, 0, 0))
            
            self.logger.info(f"Document processed successfully. Text PII: {self.audit_data['text_pii_count']}, Visual PII: {self.audit_data['visual_pii_count']}")
            
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
