"""
NLP Explanation Engine for PII Detection
Provides natural language explanations for detected PII using spaCy NER.
"""

import spacy
from typing import Dict, List, Any, Tuple
import logging
import subprocess
import sys
import os

logger = logging.getLogger(__name__)

class NLPExplain:
    """Natural Language Processing explanation engine for PII detection."""
    
    # Custom explanations for PII keywords not covered by spaCy
    CUSTOM_EXPLANATIONS = {
        "admission no": "Student admission identifier used for enrollment",
        "roll no": "Student roll number for academic identification", 
        "student id": "Unique identifier assigned to students",
        "employee id": "Unique identifier assigned to employees",
        "passport no": "Government-issued travel document number",
        "license no": "Driver's license or professional license number",
        "account no": "Bank account or financial account number",
        "policy no": "Insurance policy identification number",
        "reference no": "Reference number for transactions or documents",
        "application no": "Application reference number",
        "registration no": "Registration identifier for various services",
        "membership no": "Membership identification number",
        "ticket no": "Ticket or booking reference number",
        "invoice no": "Invoice identification number",
        "order no": "Order reference number",
        "case no": "Legal or administrative case number",
        "patient id": "Medical patient identification number",
        "medical record": "Medical record number or identifier"
    }
    
    def __init__(self):
        """Initialize the NLP explanation engine."""
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model with automatic download if missing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ])
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model downloaded and loaded successfully")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                self.nlp = None
    
    def explain_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and provide explanations for detected entities.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict containing detected entities with explanations
        """
        result = {
            "entities": [],
            "has_spacy": self.nlp is not None
        }
        
        if not text.strip():
            return result
        
        # Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                explanation = spacy.explain(ent.label_) or f"Named entity of type {ent.label_}"
                result["entities"].append({
                    "text": ent.text,
                    "label": ent.label_,
                    "explanation": explanation,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.9  # spaCy doesn't provide confidence, use default
                })
        
        # Check for custom PII keywords
        text_lower = text.lower()
        for keyword, explanation in self.CUSTOM_EXPLANATIONS.items():
            if keyword in text_lower:
                start_idx = text_lower.find(keyword)
                result["entities"].append({
                    "text": text[start_idx:start_idx + len(keyword)],
                    "label": "CUSTOM_PII",
                    "explanation": explanation,
                    "start": start_idx,
                    "end": start_idx + len(keyword),
                    "confidence": 0.8  # Lower confidence for keyword matching
                })
        
        return result
    
    def explain_bbox(self, text_fragment: str, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Analyze text fragment and attach bounding box coordinates.
        
        Args:
            text_fragment (str): Text detected in the bounding box
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (x, y, w, h)
            
        Returns:
            Dict containing explanation with bbox coordinates
        """
        explanation_data = self.explain_text(text_fragment)
        
        # Create structured result for this bbox
        result = {
            "bbox": {
                "x": bbox[0],
                "y": bbox[1], 
                "width": bbox[2],
                "height": bbox[3]
            },
            "text": text_fragment,
            "entities": explanation_data["entities"],
            "primary_explanation": self._get_primary_explanation(text_fragment, explanation_data["entities"])
        }
        
        return result
    
    def _get_primary_explanation(self, text: str, entities: List[Dict]) -> str:
        """Get the most relevant explanation for the text fragment."""
        if not entities:
            return "Text content detected as potentially sensitive"
        
        # Sort by confidence and return the best explanation
        best_entity = max(entities, key=lambda x: x.get("confidence", 0))
        return f"{best_entity['label']}: {best_entity['explanation']}"
    
    def batch_explain(self, text_fragments: List[str], bboxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Process multiple text fragments efficiently.
        
        Args:
            text_fragments (List[str]): List of text fragments
            bboxes (List[Tuple]): Corresponding bounding boxes
            
        Returns:
            List of explanation dictionaries
        """
        results = []
        for text, bbox in zip(text_fragments, bboxes):
            results.append(self.explain_bbox(text, bbox))
        return results

# Global instance for reuse
_nlp_explainer = None

def get_nlp_explainer() -> NLPExplain:
    """Get or create global NLP explainer instance."""
    global _nlp_explainer
    if _nlp_explainer is None:
        _nlp_explainer = NLPExplain()
    return _nlp_explainer