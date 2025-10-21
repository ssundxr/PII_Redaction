from typing import List, Tuple, Optional
import re
import logging
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch

PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\+?\d[\d\s\-()]{7,}\d"),
}


def detect_pii(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    findings: List[Tuple[str, Tuple[int, int]]] = []
    for label, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text):
            findings.append((label, (m.start(), m.end())))
    return findings


class VisualPIIDetector:
    """
    A visual PII detection class that uses YOLOv8 to detect potentially sensitive
    visual elements in images, such as people, faces, and other identifiable objects.
    
    This class is designed to identify visual PII that should be redacted or blurred
    in images to protect privacy and comply with data protection regulations.
    """
    
    # COCO class names for YOLOv8 (class 0 is 'person')
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    
    # Default PII-sensitive classes (primarily person detection)
    DEFAULT_PII_CLASSES = [0]  # person class
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        """
        Initialize the VisualPIIDetector with YOLOv8 model.
        
        Args:
            model_path (str): Path to the YOLOv8 model file. Defaults to "yolov8n.pt".
            confidence_threshold (float): Minimum confidence threshold for detections.
                                        Defaults to 0.25.
        
        Raises:
            Exception: If the YOLOv8 model fails to load.
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        try:
            self.logger.info(f"Loading YOLOv8 model: {model_path}")
            self.model = YOLO(model_path)
            
            # Move to GPU and use half precision if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device).half()
                
            self.logger.info(f"YOLOv8 model loaded on {self.device} with half precision")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model {model_path}: {str(e)}")
            raise Exception(f"Failed to initialize VisualPIIDetector: {str(e)}")
    
    def detect_visual_pii(self, image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Detect visual PII elements in an image using YOLOv8.
        
        This method runs YOLOv8 inference on the input image and filters detections
        for PII-sensitive classes (primarily people) with confidence above the threshold.
        
        Args:
            image (PIL.Image.Image): The input image to analyze for visual PII.
        
        Returns:
            List[Tuple[str, Tuple[int, int, int, int], float]]: A list of tuples containing:
                - class_name (str): Name of the detected class (e.g., 'person')
                - bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2)
                - confidence (float): Detection confidence score
        
        Raises:
            Exception: If visual PII detection fails.
        """
        try:
            self.logger.info("Starting visual PII detection")
            
            # Convert image to numpy array for YOLO
            img_array = np.array(image)
            
            # Run YOLOv8 inference with optimized settings
            results = self.model(
                img_array,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                agnostic_nms=True  # Better for person detection
            )
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection data
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Filter for PII-sensitive classes and confidence threshold
                        if (class_id in self.DEFAULT_PII_CLASSES and 
                            confidence >= self.confidence_threshold):
                            
                            class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
                            bbox_int = [int(coord) for coord in bbox]
                            
                            detections.append((class_name, tuple(bbox_int), confidence))
                            self.logger.debug(f"Detected {class_name} with confidence {confidence:.3f} at {bbox_int}")
            
            self.logger.info(f"Found {len(detections)} visual PII detections")
            return detections
            
        except Exception as e:
            self.logger.error(f"Visual PII detection failed: {str(e)}")
            raise Exception(f"Failed to detect visual PII: {str(e)}")
    
    def detect_custom_classes(self, image: Image.Image, 
                            target_classes: List[int], 
                            confidence_threshold: Optional[float] = None) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Detect specific object classes in an image using YOLOv8.
        
        This method allows for custom detection of specific COCO classes beyond
        the default PII-sensitive classes.
        
        Args:
            image (PIL.Image.Image): The input image to analyze.
            target_classes (List[int]): List of COCO class IDs to detect.
            confidence_threshold (Optional[float]): Override confidence threshold.
                                                  Uses instance default if None.
        
        Returns:
            List[Tuple[str, Tuple[int, int, int, int], float]]: A list of tuples containing:
                - class_name (str): Name of the detected class
                - bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2)
                - confidence (float): Detection confidence score
        
        Raises:
            Exception: If custom class detection fails.
        """
        try:
            self.logger.info(f"Starting custom class detection for classes: {target_classes}")
            
            # Use provided threshold or instance default
            threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
            
            # Run YOLOv8 inference
            results = self.model(image, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection data
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        # Filter for target classes and confidence threshold
                        if class_id in target_classes and confidence >= threshold:
                            class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
                            bbox_int = [int(coord) for coord in bbox]
                            
                            detections.append((class_name, tuple(bbox_int), confidence))
                            self.logger.debug(f"Detected {class_name} with confidence {confidence:.3f} at {bbox_int}")
            
            self.logger.info(f"Found {len(detections)} custom class detections")
            return detections
            
        except Exception as e:
            self.logger.error(f"Custom class detection failed: {str(e)}")
            raise Exception(f"Failed to detect custom classes: {str(e)}")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold for detections.
        
        Args:
            threshold (float): New confidence threshold (0.0 to 1.0).
        
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        self.logger.info(f"Updated confidence threshold to {threshold}")
    
    def get_available_classes(self) -> dict:
        """
        Get the available COCO class names and their IDs.
        
        Returns:
            dict: Dictionary mapping class IDs to class names.
        """
        return self.COCO_CLASSES.copy()
    
    def get_pii_classes(self) -> List[int]:
        """
        Get the list of default PII-sensitive class IDs.
        
        Returns:
            List[int]: List of class IDs considered PII-sensitive.
        """
        return self.DEFAULT_PII_CLASSES.copy()
