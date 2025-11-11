"""
Test PDF Redaction System
Creates a sample PDF and tests redaction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("PDF REDACTION SYSTEM TEST")
print("="*70)

# Check dependencies
print("\n[1/5] Checking PDF dependencies...")
try:
    from pdf2image import convert_from_path
    print("✓ pdf2image installed")
except ImportError:
    print("✗ pdf2image NOT installed")
    print("  Install: pip install pdf2image")
    print("  Also install poppler:")
    print("    Windows: Download from poppler.freedesktop.org")
    print("    Linux: sudo apt-get install poppler-utils")
    print("    Mac: brew install poppler")
    sys.exit(1)

try:
    import img2pdf
    print("✓ img2pdf installed")
except ImportError:
    print("✗ img2pdf NOT installed")
    print("  Install: pip install img2pdf")
    sys.exit(1)

# Import PDF redactor
print("\n[2/5] Importing PDF redactor...")
try:
    from pdf_redactor import PDFRedactor, redact_pdf_simple
    print("✓ PDF redactor imported")
except Exception as e:
    print(f"✗ Failed to import PDF redactor: {e}")
    sys.exit(1)

# Create sample PDF
print("\n[3/5] Creating sample PDF...")
try:
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    
    # Create 3-page sample PDF
    images = []
    
    for page_num in range(1, 4):
        img = Image.new('RGB', (800, 1100), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 24)
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font = ImageFont.load_default()
        
        # Add header
        draw.text((50, 50), f"Sample Document - Page {page_num}", fill='black', font=font_title)
        draw.line([(50, 90), (750, 90)], fill='black', width=2)
        
        # Add PII content
        y = 150
        if page_num == 1:
            draw.text((50, y), "Personal Information:", fill='black', font=font)
            y += 40
            draw.text((50, y), "Name: John Doe", fill='black', font=font)
            y += 30
            draw.text((50, y), "Email: john.doe@example.com", fill='black', font=font)
            y += 30
            draw.text((50, y), "Phone: +1-555-123-4567", fill='black', font=font)
            y += 30
            draw.text((50, y), "SSN: 123-45-6789", fill='black', font=font)
        
        elif page_num == 2:
            draw.text((50, y), "Account Details:", fill='black', font=font)
            y += 40
            draw.text((50, y), "Account: 9876543210123456", fill='black', font=font)
            y += 30
            draw.text((50, y), "Aadhaar: 1234 5678 9012", fill='black', font=font)
            y += 30
            draw.text((50, y), "PAN: ABCDE1234F", fill='black', font=font)
        
        elif page_num == 3:
            draw.text((50, y), "Contact Information:", fill='black', font=font)
            y += 40
            draw.text((50, y), "Customer: Jane Smith", fill='black', font=font)
            y += 30
            draw.text((50, y), "Email: jane.smith@company.com", fill='black', font=font)
            y += 30
            draw.text((50, y), "Phone: +1-555-987-6543", fill='black', font=font)
        
        # Add signature
        draw.line([(50, 900), (200, 920), (150, 940)], fill='black', width=3)
        draw.text((50, 950), "Signature", fill='gray', font=font)
        
        images.append(img)
    
    # Convert to PDF
    import img2pdf
    test_pdf = Path("output/test_sample.pdf")
    test_pdf.parent.mkdir(exist_ok=True)
    
    image_bytes = []
    for img in images:
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        image_bytes.append(img_buffer.getvalue())
    
    with open(test_pdf, 'wb') as f:
        f.write(img2pdf.convert(image_bytes))
    
    print(f"✓ Created sample PDF: {test_pdf}")
    print(f"  Pages: {len(images)}")

except Exception as e:
    print(f"✗ Failed to create sample PDF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize redactor
print("\n[4/5] Initializing PDF redactor...")
try:
    redactor = PDFRedactor()
    info = redactor.get_info()
    print(f"✓ PDF redactor initialized")
    print(f"  Max pages: {info['max_pages']}")
    print(f"  Output dir: {info['output_directory']}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# Redact PDF
print("\n[5/5] Redacting PDF...")
print("-" * 70)
try:
    output_path, audit = redactor.redact_pdf(
        str(test_pdf),
        dpi=150  # Lower DPI for faster testing
    )
    
    print("-" * 70)
    print("\n" + "="*70)
    print("REDACTION RESULTS:")
    print("="*70)
    print(f"Output: {Path(output_path).name}")
    print(f"Total Pages: {audit['total_pages']}")
    print(f"Total Detections: {audit['total_detections']}")
    print(f"Processing Time: {audit['processing_time']}s")
    print(f"\nStatistics:")
    stats = audit['statistics']
    print(f"  Pages with PII: {stats['pages_with_pii']}")
    print(f"  Pages without PII: {stats['pages_without_pii']}")
    print(f"  Avg detections/page: {stats['avg_detections_per_page']}")
    print(f"\nBy Risk:")
    for risk, count in stats['by_risk'].items():
        print(f"  {risk}: {count}")
    print(f"\nBy Type:")
    for type_, count in stats['by_type'].items():
        print(f"  {type_}: {count}")
    
    if stats['high_risk_pages']:
        print(f"\nHigh-Risk Pages: {', '.join(map(str, stats['high_risk_pages']))}")
    
    print(f"\n✅ PDF REDACTION SUCCESSFUL!")
    print(f"\nFiles created:")
    print(f"  Input:  {test_pdf}")
    print(f"  Output: {output_path}")
    print(f"  Audit:  output/pdf_audit_logs/pdf_audit_*.json")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Check the output folder for redacted PDF")
    print("2. Open the PDF to verify redactions")
    print("3. Run the GUI: python -m src.ui")
    print("4. Click 'Load PDF' to test with your own PDFs")
    print("="*70)

except Exception as e:
    print(f"\n✗ PDF REDACTION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
