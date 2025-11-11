# 📄 PDF Redaction Feature - User Guide

## ✨ New Feature: Multi-Page PDF Redaction

Your PII Redaction System now supports **PDF documents with up to 100 pages**!

---

## 🚀 Quick Start

### 1. **Install PDF Support**
```bash
pip install pdf2image img2pdf
```

### 2. **Install Poppler** (Required)

**Windows:**
```bash
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Extract to C:\Program Files\poppler
# Add to PATH: C:\Program Files\poppler\Library\bin
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**Mac:**
```bash
brew install poppler
```

### 3. **Test PDF Redaction**
```bash
python test_pdf_redaction.py
```

### 4. **Launch GUI with PDF Support**
```bash
python -m src.ui
```

---

## 📖 How to Use

### **GUI Method (Recommended)**

1. **Launch the application:**
   ```bash
   python -m src.ui
   ```

2. **Load your PDF:**
   - Click **"📄 Load PDF"** button
   - Select a PDF file (up to 100 pages)
   - The PDF name will be displayed

3. **Redact PII:**
   - Click **"🔒 Redact PII"**
   - Watch the progress: "Processing page X/Y..."
   - Wait for completion (time depends on page count)

4. **Download Redacted PDF:**
   - Click **"💾 Save Result"**
   - Choose save location
   - Your redacted PDF is ready!

### **Programmatic Method**

```python
from src.pdf_redactor import redact_pdf_simple

# Simple one-liner
output_path = redact_pdf_simple("input.pdf")
print(f"Redacted PDF: {output_path}")
```

**Advanced usage:**
```python
from src.pdf_redactor import PDFRedactor

# Initialize
redactor = PDFRedactor()

# Redact with progress callback
def progress(page, total, status):
    print(f"[{page}/{total}] {status}")

output_path, audit = redactor.redact_pdf(
    "input.pdf",
    output_filename="my_redacted.pdf",
    dpi=200,
    progress_callback=progress
)

# Check statistics
print(f"Total detections: {audit['total_detections']}")
print(f"Processing time: {audit['processing_time']}s")
```

---

## 📊 What Gets Redacted

The PDF redactor detects and redacts the same PII as image mode:

### **Text PII**
- ✅ Email addresses
- ✅ Phone numbers
- ✅ SSN (Social Security Numbers)
- ✅ Aadhaar numbers
- ✅ PAN cards
- ✅ Credit card numbers
- ✅ Passport numbers
- ✅ Names (context-aware)
- ✅ Account numbers

### **Visual PII**
- ✅ Signatures
- ✅ Photos/Faces
- ✅ QR codes
- ✅ Stamps/Seals

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Multi-page Support** | Process up to 100 pages in one go |
| **Page-by-Page Processing** | Each page redacted individually |
| **Progress Tracking** | Real-time status: "Processing page X/Y" |
| **Comprehensive Audit** | Detailed statistics per page |
| **High Quality** | 200 DPI output (adjustable) |
| **Fast Processing** | ~0.5-2 seconds per page |
| **Automatic Download** | Redacted PDF ready to download |

---

## 📈 Processing Times

| Pages | Estimated Time | Output Size |
|-------|----------------|-------------|
| 1-5 pages | 5-15 seconds | 500KB-2MB |
| 10-20 pages | 20-40 seconds | 2-5MB |
| 50 pages | 1-2 minutes | 10-15MB |
| 100 pages | 3-5 minutes | 20-30MB |

*Times vary based on system specs and page complexity*

---

## 📁 Output Structure

```
output/
├── pdf_redacted/
│   └── filename_redacted_20241107_143052.pdf  ← Redacted PDF
└── pdf_audit_logs/
    └── pdf_audit_1cc473d1f230.json            ← Detailed audit
```

---

## 📋 Audit Log Example

```json
{
  "filename": "document.pdf",
  "timestamp": "2024-11-07T14:30:52",
  "total_pages": 3,
  "total_detections": 15,
  "processing_time": 12.45,
  "pages": [
    {
      "page_number": 1,
      "detections": 5,
      "statistics": {
        "by_risk": {"HIGH": 3, "MEDIUM": 2},
        "by_type": {"PATTERN": 3, "NAME": 2}
      },
      "risk_assessment": "HIGH - High-risk PII present"
    },
    ...
  ],
  "statistics": {
    "total_pages": 3,
    "pages_with_pii": 3,
    "pages_without_pii": 0,
    "avg_detections_per_page": 5.0,
    "by_risk": {"HIGH": 8, "MEDIUM": 7},
    "high_risk_pages": [1, 2]
  }
}
```

---

## ⚙️ Configuration

### **Adjust DPI (Quality vs Speed)**

Edit `src/pdf_redactor.py` line 67:
```python
def redact_pdf(self, pdf_path: str, dpi: int = 200):  # Change 200 to 150 for faster
```

| DPI | Quality | Speed | File Size |
|-----|---------|-------|-----------|
| 150 | Good | Fast | Smaller |
| 200 | Better | Medium | Medium |
| 300 | Best | Slow | Larger |

### **Adjust Page Limit**

Edit `src/pdf_redactor.py` line 42:
```python
MAX_PAGES = 100  # Change to increase/decrease limit
```

---

## 🔧 Troubleshooting

### **Error: "pdf2image not available"**
```bash
pip install pdf2image
```

### **Error: "Unable to get page count"**
- **Cause**: Poppler not installed
- **Fix**: Follow installation steps in [PDF_SETUP.md](PDF_SETUP.md)

### **Error: "PDF has X pages. Maximum allowed: 100"**
- **Cause**: PDF exceeds page limit
- **Fix**: Split PDF or increase MAX_PAGES in `pdf_redactor.py`

### **Slow Processing**
- **Reduce DPI**: Change from 200 to 150
- **Check System**: Ensure sufficient RAM (8GB+ recommended)

### **"Memory Error"**
- **Solution**: Process smaller PDFs or increase system RAM
- **Alternative**: Split PDF into smaller chunks

---

## 💡 Tips & Best Practices

1. **Test First**: Always test with `test_pdf_redaction.py` before production use
2. **Backup Originals**: Keep original PDFs before redacting
3. **Review Output**: Always review redacted PDF to ensure accuracy
4. **Batch Processing**: For multiple PDFs, process one at a time for stability
5. **Quality vs Speed**: Use 200 DPI for production, 150 for testing

---

## 🎓 Technical Details

### **Processing Pipeline**

1. **PDF → Images**
   - Converts each page to high-res image (200 DPI)
   - Uses poppler for accurate rendering

2. **PII Detection**
   - OCR extracts text from each page
   - Pattern matching finds PII (email, phone, SSN, etc.)
   - Computer vision detects signatures, faces, QR codes

3. **Redaction**
   - Black boxes drawn over detected PII
   - Preserves document layout

4. **Images → PDF**
   - Converts redacted images back to PDF
   - Maintains page order and quality

### **Technologies Used**

- **pdf2image**: PDF to image conversion (via poppler)
- **img2pdf**: Image to PDF conversion
- **Tesseract OCR**: Text extraction
- **OpenCV**: Computer vision for signatures/QR codes
- **YOLO**: Face/photo detection
- **Regex**: Pattern-based PII detection

---

## 📊 Comparison: Image vs PDF Mode

| Feature | Image Mode | PDF Mode |
|---------|------------|----------|
| Input | Single image | Multi-page PDF |
| Max Pages | 1 | 100 |
| Output | Image file | PDF file |
| Audit Log | Single page | Per-page breakdown |
| Processing | Instant | 0.5-2s per page |
| Use Case | Quick test | Production docs |

---

## 🚀 Next Steps

1. **Install dependencies** (see above)
2. **Run test**: `python test_pdf_redaction.py`
3. **Try your PDFs** in the GUI
4. **Review audit logs** for detailed statistics
5. **Integrate** into your workflow

---

## 📞 Support

For issues or questions:
1. Check [PDF_SETUP.md](PDF_SETUP.md) for installation help
2. Run diagnostic: `python test_pdf_redaction.py`
3. Check logs in `output/pdf_audit_logs/`

---

## ✅ Summary

**You can now:**
- ✅ Upload PDF files (up to 100 pages)
- ✅ Redact ALL PII automatically
- ✅ Download redacted PDF
- ✅ Get detailed audit logs
- ✅ Process in GUI or programmatically

**Ready to redact PDFs! 🎉**
