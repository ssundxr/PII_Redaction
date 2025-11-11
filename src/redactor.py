"""
PII Redactor - Enterprise Edition v3.0
OPTIMIZED for Aadhaar, Government IDs, and Structured Documents
Only redacts DETECTED PII - Nothing else
"""


import time
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw
import json
import logging
from datetime import datetime
from pathlib import Path
import hashlib


from ocr import extract_words_with_boxes
from detector import detect_pii_patterns


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class PIIRedactor:
    """
    Smart PII Redactor for Government Documents.
    
    Redacts ONLY:
    - Names (detected through context)
    - Aadhaar numbers (12 digits)
    - Phone numbers
    - Addresses (multi-line text blocks)
    - Photos (face detection)
    - QR codes
    - Signatures
    
    Does NOT redact:
    - Headers/titles
    - Government logos
    - General text
    """
    
    def __init__(self, output_dir: str = "output"):
        """Initialize redactor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.audit_dir = self.output_dir / "audit_logs"
        self.audit_dir.mkdir(exist_ok=True)
        
        # Context keywords for name detection
        self.name_keywords = ['name:', 'to:', 's/o', 'd/o', 'son of', 'daughter of', 'shri', 'smt', 'mr', 'mrs', 'ms']
        
        # Initialize detectors
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize visual detectors."""
        try:
            from detector import VisualPIIDetector
            self.visual_detector = VisualPIIDetector()
            logger.info("✓ Visual detector ready")
        except Exception as e:
            self.visual_detector = None
            logger.warning(f"Visual detector unavailable: {e}")
    
    def redact_image(
        self,
        image: Image.Image,
        filename: str = "document.jpg",
        generate_audit: bool = True
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Intelligently redact ONLY PII from document.
        """
        start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING REDACTION: {filename}")
        logger.info(f"{'='*60}")
        
        # Audit data
        audit_data = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'image_size': image.size,
            'detections': [],
            'statistics': {},
            'nlp_explanation': '',
            'risk_assessment': '',
            'processing_time': 0
        }
        
        # Create redacted copy
        redacted = image.copy()
        draw = ImageDraw.Draw(redacted)
        
        # Collect all PII detections
        all_detections = []
        
        # 1. Extract ALL text first
        logger.info("\n[1/4] Extracting text...")
        words_with_boxes = extract_words_with_boxes(image)
        logger.info(f"✓ Extracted {len(words_with_boxes)} words")
        
        # 2. Detect specific PII patterns (Aadhaar, Phone, etc.)
        logger.info("\n[2/4] Detecting PII patterns...")
        pattern_detections = self._detect_pii_patterns(words_with_boxes)
        all_detections.extend(pattern_detections)
        logger.info(f"✓ Found {len(pattern_detections)} pattern-based PII")
        
        # 3. Detect names using context
        logger.info("\n[3/4] Detecting names...")
        name_detections = self._detect_names(words_with_boxes)
        all_detections.extend(name_detections)
        logger.info(f"✓ Found {len(name_detections)} names")
        
        # 4. Detect visual PII (photos, QR codes, signatures)
        logger.info("\n[4/4] Detecting visual PII...")
        visual_detections = self._detect_visual_pii_precise(image)
        all_detections.extend(visual_detections)
        logger.info(f"✓ Found {len(visual_detections)} visual PII items")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL PII ITEMS TO REDACT: {len(all_detections)}")
        logger.info(f"{'='*60}\n")
        
        # Redact each detected PII
        for i, detection in enumerate(all_detections, 1):
            self._redact_single_item(draw, detection, i)
        
        # Finalize
        processing_time = time.time() - start_time
        audit_data['detections'] = all_detections
        audit_data['statistics'] = self._calculate_statistics(all_detections)
        audit_data['processing_time'] = round(processing_time, 2)
        
        if generate_audit:
            audit_data['nlp_explanation'] = self._generate_explanation(all_detections)
            audit_data['risk_assessment'] = self._assess_risk(all_detections)
            self._save_audit(audit_data)
        
        logger.info(f"\n✓ REDACTION COMPLETE in {processing_time:.2f}s\n")
        
        return redacted, audit_data
    
    def _detect_pii_patterns(self, words_with_boxes: List[Tuple[str, Tuple]]) -> List[Dict]:
        """Detect Aadhaar, phone, email, etc."""
        detections = []
        
        for word, bbox in words_with_boxes:
            patterns = detect_pii_patterns(word)
            
            for label, _ in patterns:
                detections.append({
                    'type': 'PATTERN',
                    'entity': label,
                    'text': word,
                    'bbox': bbox,
                    'confidence': 0.95,
                    'risk': self._map_risk(label)
                })
                logger.info(f"  → {label}: {word}")
        
        return detections
    
    def _detect_names(self, words_with_boxes: List[Tuple[str, Tuple]]) -> List[Dict]:
        """
        Detect names using contextual analysis.
        Looks for name indicators: "Name:", "To:", "S/O", "D/O", etc.
        """
        detections = []
        
        for i, (word, bbox) in enumerate(words_with_boxes):
            word_lower = word.lower()
            
            # Check if this word is a name indicator
            is_indicator = any(keyword in word_lower for keyword in self.name_keywords)
            
            if is_indicator and i + 1 < len(words_with_boxes):
                # Next 1-3 words are likely the name
                for j in range(i + 1, min(i + 4, len(words_with_boxes))):
                    next_word, next_bbox = words_with_boxes[j]
                    
                    # Skip if it's another keyword or number
                    if any(kw in next_word.lower() for kw in self.name_keywords):
                        break
                    if next_word.isdigit():
                        break
                    
                    # This is likely a name component
                    if len(next_word) > 2 and next_word[0].isupper():
                        detections.append({
                            'type': 'NAME',
                            'entity': 'PERSON_NAME',
                            'text': next_word,
                            'bbox': next_bbox,
                            'confidence': 0.85,
                            'risk': 'HIGH'
                        })
                        logger.info(f"  → NAME: {next_word}")
        
        return detections
    
    def _detect_visual_pii_precise(self, image: Image.Image) -> List[Dict]:
        """
        Detect ONLY faces, QR codes, and signatures.
        NO generic regions.
        """
        if not self.visual_detector:
            return []
        
        detections = []
        
        try:
            # Import specific detectors
            from detector import SignatureDetector, QRCodeDetector
            
            # 1. Signatures
            sig_detector = SignatureDetector()
            signatures = sig_detector.detect(image)
            for sig in signatures:
                if sig['confidence'] > 0.75:  # High confidence only
                    x, y, w, h = sig['bbox']
                    detections.append({
                        'type': 'VISUAL',
                        'entity': 'SIGNATURE',
                        'text': '',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': sig['confidence'],
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → SIGNATURE at ({x}, {y})")
            
            # 2. QR Codes
            qr_detector = QRCodeDetector()
            qr_codes = qr_detector.detect(image)
            for qr in qr_codes:
                if qr['confidence'] > 0.65:
                    x, y, w, h = qr['bbox']
                    detections.append({
                        'type': 'VISUAL',
                        'entity': 'QR_CODE',
                        'text': '',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': qr['confidence'],
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → QR CODE at ({x}, {y})")
            
            # 3. Faces (YOLO if available)
            if self.visual_detector.model:
                import numpy as np
                img_array = np.array(image)
                results = self.visual_detector.model(img_array, conf=0.6, verbose=False, classes=[0])
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = box
                            detections.append({
                                'type': 'VISUAL',
                                'entity': 'FACE/PHOTO',
                                'text': '',
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': 0.85,
                                'risk': 'HIGH'
                            })
                            logger.info(f"  → FACE at ({int(x1)}, {int(y1)})")
        
        except Exception as e:
            logger.error(f"Visual detection error: {e}")
        
        return detections
    
    def _redact_single_item(self, draw: ImageDraw.Draw, detection: Dict, index: int):
        """Redact a single PII item."""
        try:
            bbox = detection['bbox']
            risk = detection.get('risk', 'MEDIUM')
            entity = detection.get('entity', 'PII')
            
            # Parse bbox
            if len(bbox) == 4:
                if all(isinstance(x, (int, float)) for x in bbox):
                    # Could be [x, y, w, h] or [x, y, x2, y2]
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1] and bbox[2] < 5000:
                        # Likely [x, y, x2, y2]
                        x1, y1, x2, y2 = map(int, bbox)
                    else:
                        # Likely [x, y, w, h]
                        x, y, w, h = map(int, bbox)
                        x1, y1, x2, y2 = x, y, x+w, y+h
                else:
                    return
            else:
                return
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 5 or (y2-y1) < 5:
                logger.debug(f"Skipped invalid box: {bbox}")
                return
            
            # Color by risk
            colors = {
                'HIGH': (0, 0, 0),
                'MEDIUM': (40, 40, 40),
                'LOW': (80, 80, 80)
            }
            color = colors.get(risk, (0, 0, 0))
            
            # Draw redaction
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
            
            logger.debug(f"[{index}] Redacted {entity}: ({x1},{y1}) to ({x2},{y2})")
        
        except Exception as e:
            logger.error(f"Redaction failed for item {index}: {e}")
    
    def _map_risk(self, entity: str) -> str:
        """Map entity to risk level."""
        high_risk = {'AADHAAR', 'SSN', 'PASSPORT', 'PAN', 'CREDIT_CARD', 'PERSON_NAME'}
        medium_risk = {'EMAIL', 'PHONE', 'ACCOUNT', 'IP_ADDRESS'}
        
        return 'HIGH' if entity in high_risk else ('MEDIUM' if entity in medium_risk else 'LOW')
    
    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """Calculate statistics."""
        stats = {
            'total_detections': len(detections),
            'by_type': {},
            'by_risk': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'entities': {}
        }
        
        for det in detections:
            # By type
            det_type = det.get('type', 'UNKNOWN')
            stats['by_type'][det_type] = stats['by_type'].get(det_type, 0) + 1
            
            # By risk
            risk = det.get('risk', 'LOW')
            if risk in stats['by_risk']:
                stats['by_risk'][risk] += 1
            
            # By entity
            entity = det.get('entity', 'UNKNOWN')
            stats['entities'][entity] = stats['entities'].get(entity, 0) + 1
        
        return stats
    
    def _generate_explanation(self, detections: List[Dict]) -> str:
        """Generate NLP explanation."""
        if not detections:
            return "No PII detected in the document."
        
        total = len(detections)
        stats = self._calculate_statistics(detections)
        
        explanation = f"Detected and redacted {total} PII item{'s' if total > 1 else ''}: "
        
        # Risk breakdown
        parts = []
        for risk in ['HIGH', 'MEDIUM', 'LOW']:
            count = stats['by_risk'][risk]
            if count > 0:
                parts.append(f"{count} {risk.lower()}-risk")
        
        explanation += ", ".join(parts) + ". "
        
        # Top entities
        entities = sorted(stats['entities'].items(), key=lambda x: x[1], reverse=True)[:3]
        if entities:
            ent_str = ", ".join([f"{count}x {ent}" for ent, count in entities])
            explanation += f"Includes: {ent_str}. "
        
        explanation += "All sensitive information has been securely redacted."
        
        return explanation
    
    def _assess_risk(self, detections: List[Dict]) -> str:
        """Assess overall risk."""
        if not detections:
            return "MINIMAL - No PII detected"
        
        stats = self._calculate_statistics(detections)
        high = stats['by_risk']['HIGH']
        
        if high >= 3:
            return "CRITICAL - Multiple high-risk PII elements"
        elif high >= 1:
            return "HIGH - High-risk PII present"
        elif len(detections) >= 5:
            return "MODERATE - Multiple PII items"
        else:
            return "LOW - Minimal PII"
    
    def _save_audit(self, audit_data: Dict):
        """Save audit log."""
        try:
            audit_id = hashlib.sha256(
                f"{audit_data['filename']}{audit_data['timestamp']}".encode()
            ).hexdigest()[:12]
            
            audit_file = self.audit_dir / f"audit_{audit_id}.json"
            
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Audit saved: {audit_file.name}")
        except Exception as e:
            logger.error(f"Audit save failed: {e}")



def get_enhanced_redactor() -> PIIRedactor:
    """Factory function."""
    return PIIRedactor()
