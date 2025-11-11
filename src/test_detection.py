"""Test PII detection"""
from PIL import Image, ImageDraw, ImageFont
from redactor import PIIRedactor

# Create a test image with PII
img = Image.new('RGB', (800, 400), 'white')
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 30)
except:
    font = ImageFont.load_default()

# Add PII text
draw.text((50, 50), "Email: john.doe@example.com", fill='black', font=font)
draw.text((50, 100), "Phone: +1-555-123-4567", fill='black', font=font)
draw.text((50, 150), "SSN: 123-45-6789", fill='black', font=font)
draw.text((50, 200), "Aadhaar: 1234 5678 9012", fill='black', font=font)
draw.text((50, 250), "Account: 9876543210123", fill='black', font=font)

# Save test image
img.save("test4.png")
print("✓ Created test image: test4.png")

# Test redaction
print("\nTesting redaction...")
redactor = PIIRedactor()
redacted, audit = redactor.redact_image(img, "test_document.png")

# Save redacted
redacted.save("test_pii_REDACTED.png")
print(f"✓ Redacted image saved: test_pii_REDACTED.png")

# Print audit
print(f"\n📊 Detection Summary:")
print(f"  Total detections: {audit['statistics']['total_detections']}")
print(f"  By risk: {audit['statistics']['by_risk']}")
print(f"  Processing time: {audit['processing_time']}s")
print(f"\n💬 NLP Explanation:")
print(f"  {audit['nlp_explanation']}")
