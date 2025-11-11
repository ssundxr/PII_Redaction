"""
Complete PII Redaction System Test
Diagnoses and tests ALL components
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("PII REDACTION SYSTEM - COMPLETE DIAGNOSTIC TEST")
print("="*70)

# ============================================
# PHASE 1: DEPENDENCY CHECK
# ============================================
print("\n[PHASE 1] CHECKING DEPENDENCIES...")
print("-"*70)

dependencies = {
    'cv2': 'OpenCV',
    'torch': 'PyTorch',
    'PIL': 'Pillow',
    'pytesseract': 'Tesseract',
    'transformers': 'Transformers',
    'ultralytics': 'YOLO'
}

missing = []
available = []

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name:20} - INSTALLED")
        available.append(name)
    except ImportError:
        print(f"✗ {name:20} - MISSING")
        missing.append(name)

if missing:
    print(f"\n⚠ MISSING: {', '.join(missing)}")
    print("Install with: pip install opencv-python torch pillow pytesseract transformers ultralytics")
else:
    print("\n✓ All dependencies installed!")

# ============================================
# PHASE 2: TESSERACT CHECK
# ============================================
print("\n[PHASE 2] CHECKING TESSERACT OCR...")
print("-"*70)

try:
    import pytesseract
    
    # Try to get version
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract Version: {version}")
    except Exception as e:
        print(f"✗ Tesseract executable not found: {e}")
        print("\nPossible fixes:")
        print("1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Update path in src/ocr.py line 16")
        
except ImportError:
    print("✗ pytesseract package not installed")

# ============================================
# PHASE 3: MODULE IMPORTS
# ============================================
print("\n[PHASE 3] IMPORTING PII MODULES...")
print("-"*70)

modules = {}

try:
    from detector import VisualPIIDetector, detect_pii_patterns, SignatureDetector, QRCodeDetector
    print("✓ detector.py imported")
    modules['detector'] = True
except Exception as e:
    print(f"✗ detector.py failed: {e}")
    modules['detector'] = False

try:
    from ocr import extract_text, extract_words_with_boxes, TrOCRExtractor
    print("✓ ocr.py imported")
    modules['ocr'] = True
except Exception as e:
    print(f"✗ ocr.py failed: {e}")
    modules['ocr'] = False

try:
    from layoutlm_detector import LayoutLMDetector
    print("✓ layoutlm_detector.py imported")
    modules['layoutlm'] = True
except Exception as e:
    print(f"✗ layoutlm_detector.py failed: {e}")
    modules['layoutlm'] = False

try:
    from redactor import PIIRedactor
    print("✓ redactor.py imported")
    modules['redactor'] = True
except Exception as e:
    print(f"✗ redactor.py failed: {e}")
    modules['redactor'] = False

if not all(modules.values()):
    print("\n⚠ Some modules failed to import!")
    sys.exit(1)

# ============================================
# PHASE 4: PATTERN DETECTION TEST
# ============================================
print("\n[PHASE 4] TESTING PATTERN DETECTION...")
print("-"*70)

test_text = """
Name: John Doe
Email: john.doe@example.com
Phone: +1-555-123-4567
Aadhaar: 1234 5678 9012
PAN: ABCDE1234F
SSN: 123-45-6789
"""

patterns = detect_pii_patterns(test_text)
print(f"Detected {len(patterns)} PII patterns:")
for label, (start, end) in patterns:
    print(f"  → {label}: {test_text[start:end]}")

if len(patterns) == 0:
    print("⚠ WARNING: No patterns detected!")
else:
    print("✓ Pattern detection working")

# ============================================
# PHASE 5: OCR TEST
# ============================================
print("\n[PHASE 5] TESTING OCR...")
print("-"*70)

try:
    from PIL import Image, ImageDraw, ImageFont
    
    # Create test image
    img = Image.new('RGB', (400, 200), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((20, 20), "Name: John Doe", fill='black', font=font)
    draw.text((20, 60), "Email: test@example.com", fill='black', font=font)
    draw.text((20, 100), "Phone: 555-1234", fill='black', font=font)
    
    # Test OCR
    text = extract_text(img)
    print(f"Extracted text ({len(text)} chars):")
    print(f"  {text[:100]}...")
    
    if len(text) > 10:
        print("✓ OCR working")
    else:
        print("⚠ WARNING: OCR returned very little text")
    
    # Test word boxes
    words = extract_words_with_boxes(img)
    print(f"Extracted {len(words)} words with boxes")
    
    if len(words) > 0:
        print("✓ Word extraction working")
    else:
        print("⚠ WARNING: No words extracted")
        
except Exception as e:
    print(f"✗ OCR test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# PHASE 6: VISUAL DETECTOR TEST
# ============================================
print("\n[PHASE 6] TESTING VISUAL DETECTORS...")
print("-"*70)

try:
    # Test signature detector
    sig_detector = SignatureDetector()
    print("✓ SignatureDetector initialized")
    
    # Test QR detector
    qr_detector = QRCodeDetector()
    print("✓ QRCodeDetector initialized")
    
    # Test main visual detector
    visual_detector = VisualPIIDetector()
    print("✓ VisualPIIDetector initialized")
    print(f"  Device: {visual_detector.device}")
    print(f"  YOLO loaded: {visual_detector.model is not None}")
    
except Exception as e:
    print(f"✗ Visual detector test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# PHASE 7: FULL REDACTION TEST
# ============================================
print("\n[PHASE 7] TESTING FULL REDACTION...")
print("-"*70)

try:
    # Create test document
    test_img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(test_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add PII
    draw.text((50, 50), "Name: John Doe", fill='black', font=font)
    draw.text((50, 100), "Email: john@example.com", fill='black', font=font)
    draw.text((50, 150), "Phone: +1-555-1234", fill='black', font=font)
    draw.text((50, 200), "Aadhaar: 1234 5678 9012", fill='black', font=font)
    
    # Add signature simulation
    draw.line([(50, 300), (200, 320), (150, 340)], fill='black', width=3)
    
    print("Created test document")
    
    # Initialize redactor
    redactor = PIIRedactor()
    print("✓ PIIRedactor initialized")
    
    # Redact
    print("\nProcessing document...")
    redacted_img, audit = redactor.redact_image(test_img, filename="test.jpg", generate_audit=True)
    
    print("\n" + "="*70)
    print("REDACTION RESULTS:")
    print("="*70)
    print(f"Processing time: {audit['processing_time']}s")
    print(f"Total detections: {audit['statistics']['total_detections']}")
    print(f"\nBy risk:")
    for risk, count in audit['statistics']['by_risk'].items():
        print(f"  {risk}: {count}")
    print(f"\nBy type:")
    for type_, count in audit['statistics']['by_type'].items():
        print(f"  {type_}: {count}")
    print(f"\nTop entities:")
    for entity, count in list(audit['statistics']['entities'].items())[:5]:
        print(f"  {entity}: {count}")
    
    # Save test results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    test_img.save(output_dir / "test_original.png")
    redacted_img.save(output_dir / "test_redacted.png")
    
    print(f"\n✓ Test images saved to output/")
    print(f"  - test_original.png")
    print(f"  - test_redacted.png")
    
    if audit['statistics']['total_detections'] > 0:
        print("\n✓✓✓ SYSTEM FULLY FUNCTIONAL ✓✓✓")
    else:
        print("\n⚠ WARNING: No PII detected - check configuration")
        
except Exception as e:
    print(f"\n✗ REDACTION TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

if missing:
    print(f"❌ Missing dependencies: {', '.join(missing)}")
    print("   Run: pip install " + " ".join(missing).lower().replace(" ", "-"))

if not all(modules.values()):
    print("❌ Some modules failed to import")

print("\nNext steps:")
print("1. Fix any errors shown above")
print("2. Install missing dependencies")
print("3. Configure Tesseract path in src/ocr.py if needed")
print("4. Run: python -m src.ui")

print("\n" + "="*70)
