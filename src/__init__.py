"""
PII Redaction System - Auto-setup and initialization
"""

import subprocess
import sys
import logging

__all__ = ["ocr", "detector", "redactor", "ui"]

def _setup_spacy_model():
    """Auto-download spaCy model if not present."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        try:
            print("Installing spaCy English model...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            print("spaCy model installed successfully!")
        except Exception as e:
            logging.warning(f"Could not install spaCy model: {e}")

# Auto-setup on import
try:
    _setup_spacy_model()
except Exception:
    pass  # Fail silently to avoid breaking imports
