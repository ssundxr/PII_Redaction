# 🛡️ Privara Intellectus - PII Redaction System# PII Redaction System



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)A modular Python project for detecting and redacting Personally Identifiable Information (PII) from images and text. It provides OCR to extract text from images, regex-based PII detection, and utilities to redact detected spans.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)## Features

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

- OCR via `pytesseract` to extract text and word bounding boxes.

**AI-Powered Multi-Modal PII Detection and Redaction Platform**- Simple regex-based PII detectors (e.g., emails, phone numbers).

- Text redaction and image box redaction utilities.

Privara Intellectus is an enterprise-grade system that automatically detects and redacts Personally Identifiable Information (PII) from documents using cutting-edge computer vision, NLP, and deep learning techniques.- Extensible structure for plugging in advanced AI models under `models/`.



---## Project Structure



## 🌟 Key Features```

pii-redaction-system/

### 🔍 **Multi-Modal PII Detection**├─ src/

- **Pattern-Based Detection**: Email, Phone, SSN, Aadhaar, PAN, Credit Cards, Medical IDs│  ├─ __init__.py

- **Visual Detection**: Signatures (3 methods), QR codes, Faces (YOLO v8)│  ├─ ocr.py

- **NLP-Based Detection**: Name extraction using BioBERT and context analysis│  ├─ detector.py

- **Structured Document Analysis**: LayoutLM v3 integration for form understanding│  ├─ redactor.py

│  └─ ui.py

### 📄 **Document Support**├─ models/

- **Images**: JPG, PNG (high-resolution support)├─ tests/

- **PDF**: Multi-page processing (up to 100 pages)│  └─ sample_images/

- **Batch Processing**: Process multiple documents simultaneously├─ .gitignore

├─ requirements.txt

### 🎨 **Professional UI**└─ README.md

- **Modern Dark Theme**: Enterprise-grade interface with smooth animations```

- **Real-Time Statistics**: Interactive dashboard with detection metrics

- **Before/After Comparison**: Toggle between original and redacted views## Setup

- **Drag & Drop**: Intuitive file upload experience

- Python 3.9+

### 📊 **Audit & Compliance**- Tesseract OCR installed and available on PATH (or set the `pytesseract.pytesseract.tesseract_cmd`).

- **Comprehensive Logging**: JSON audit trails with timestamps

- **Risk Assessment**: HIGH/MEDIUM/LOW risk classificationInstall dependencies:

- **Export Capabilities**: Download audit logs and redacted documents

- **Processing Analytics**: Detailed statistics and performance metrics```bash

pip install -r requirements.txt

---```



## 🚀 Quick Start Guide## Usage



### **Prerequisites**Run the simple CLI that performs OCR on an image, detects PII in the text, and prints the redacted text:



1. **Python 3.8 or higher**```bash

   ```bashpython -m src.ui path/to/image.png

   python --version```

   ```

## Notes

2. **Tesseract OCR** (Required for text extraction)

   - Download: https://github.com/UB-Mannheim/tesseract/wiki- Place large or custom AI models under `models/` (ignored by git by default).

   - Install to default location or update path in `src/ocr.py`- The detectors in `src/detector.py` are extensible. You can add patterns or integrate ML models as needed.


3. **Poppler** (Required for PDF processing)
   - Download: https://github.com/oschwartz10612/poppler-windows/releases/
   - Extract to project root as `poppler-25.07.0/` or system PATH

---

## 📦 Installation

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/ssundxr/PII_Redaction.git
cd PII_Redaction
```

### **Step 2: Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `opencv-python` - Computer vision operations
- `torch` & `torchvision` - Deep learning framework
- `transformers` - Hugging Face models (LayoutLM, TrOCR, BioBERT)
- `pytesseract` - OCR interface
- `pdf2image` - PDF to image conversion
- `img2pdf` - Image to PDF conversion
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `pyzbar` - QR code detection

### **Step 4: Download YOLO Models**
```bash
# Download YOLOv8 nano model for face detection
# Place in project root: yolov8n.pt
```
Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

---

## 🎯 Usage Guide

### **Option 1: Modern UI (Recommended)**

Launch the professional enterprise UI:

```bash
python src/ui_modern.py
```

**Features:**
- 🖼️ Load images or PDFs via drag & drop or file browser
- 🔒 Click "Redact PII" to process documents
- 📊 View real-time statistics and detection metrics
- 💾 Save redacted documents and export audit logs
- 🔄 Toggle between original and redacted views

---

### **Option 2: Classic UI**

Launch the traditional Tkinter interface:

```bash
python src/ui.py
```

---

### **Option 3: Command Line (Advanced)**

#### **Process a Single Image:**

```python
from src.redactor import PIIRedactor
import cv2

# Initialize redactor
redactor = PIIRedactor()

# Load image
image = cv2.imread('path/to/document.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redact PII
result = redactor.redact_image(image_rgb)

# Save results
cv2.imwrite('redacted.jpg', cv2.cvtColor(result['redacted_image'], cv2.COLOR_RGB2BGR))
print(f"Found {len(result['detections'])} PII items")
```

#### **Process a PDF:**

```python
from src.pdf_redactor import PDFRedactor

# Initialize PDF redactor
pdf_redactor = PDFRedactor()

# Process PDF
result = pdf_redactor.redact_pdf('path/to/document.pdf')

# Results
print(f"Processed {result['summary']['total_pages']} pages")
print(f"Found {result['summary']['total_pii_found']} PII items")
print(f"Output: {result['output_pdf_path']}")
```

---

## 🧪 Testing

### **Run System Validation:**
```bash
python test_complete_system.py
```
Validates all components and dependencies.

### **Test PDF Redaction:**
```bash
python test_pdf_redaction.py
```
Tests multi-page PDF processing with sample document.

### **Test Individual Components:**
```bash
python test_privara_system.py
```
Tests detector, OCR, and redactor modules individually.

---

## 📁 Project Structure

```
PII_Redaction/
├── src/
│   ├── detector.py              # Visual PII detection (signatures, QR, faces)
│   ├── ocr.py                   # Text extraction (Tesseract + TrOCR)
│   ├── layoutlm_detector.py     # Structured document understanding
│   ├── redactor.py              # Main orchestrator
│   ├── pdf_redactor.py          # PDF processing engine
│   ├── ui.py                    # Classic Tkinter UI
│   ├── ui_modern.py             # Modern professional UI ⭐
│   ├── privacy/
│   │   └── nlp_explain.py       # NLP-based PII explanation
│   └── storage/
│       └── local_db.py          # Audit log storage
├── output/
│   ├── audit_logs/              # JSON audit trails
│   ├── pdf_redacted/            # Redacted PDFs
│   └── pdf_audit_logs/          # PDF-specific audits
├── tests/
│   └── sample_images/           # Test images
├── test_complete_system.py      # Full system test
├── test_pdf_redaction.py        # PDF test suite
├── requirements.txt             # Python dependencies
├── privacy_config.json          # Configuration settings
└── README.md                    # This file
```

---

## 🔧 Configuration

### **Privacy Settings** (`privacy_config.json`)

```json
{
  "redaction_color": [0, 0, 0],
  "redaction_opacity": 1.0,
  "min_confidence": 0.5,
  "enable_visual_detection": true,
  "enable_nlp_detection": true,
  "enable_pattern_detection": true,
  "audit_logging": true
}
```

### **Custom Tesseract Path**

If Tesseract is not in PATH, update `src/ocr.py`:
```python
# Line ~20
pytesseract.pytesseract.tesseract_cmd = r'C:\Your\Path\To\tesseract.exe'
```

### **Custom Poppler Path**

If Poppler is not in PATH, update `src/pdf_redactor.py`:
```python
# Line ~50
self.poppler_path = r'C:\Your\Path\To\poppler\Library\bin'
```

---

## 📊 Detection Capabilities

### **Pattern-Based Detection**

| Type | Pattern | Example |
|------|---------|---------|
| Email | RFC 5322 compliant | `user@example.com` |
| Phone | International formats | `+1-555-123-4567` |
| SSN | US Social Security | `123-45-6789` |
| Aadhaar | Indian 12-digit ID | `1234 5678 9012` |
| PAN | Indian tax ID | `ABCDE1234F` |
| Credit Card | Luhn algorithm | `4532-1234-5678-9010` |

### **Visual Detection**

- **Signatures**: 
  - Stroke analysis (contour detection)
  - Isolated region detection (connected components)
  - Handwriting pattern recognition (aspect ratio + density)

- **QR Codes**: 
  - Blob detection (finder patterns)
  - Grid structure analysis

- **Faces**: 
  - YOLO v8 object detection
  - Confidence threshold: 0.3

### **NLP Detection**

- **Name Extraction**:
  - BioBERT entity recognition
  - Context-aware filtering (keywords: Name, Patient, Doctor, etc.)
  - Capitalization pattern analysis

---

## 🎨 UI Screenshots

### Modern UI Dashboard
- **Dark Theme**: Professional enterprise design
- **Real-Time Stats**: Live detection metrics
- **Interactive Preview**: Zoom, pan, compare views

### Processing Flow
1. **Upload** → Drag & drop or file browser
2. **Process** → AI-powered PII detection
3. **Review** → Inspect detections and statistics
4. **Export** → Save redacted documents + audit logs

---

## 📈 Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Image Processing | ~0.4-0.8s | 1920x1080 resolution |
| PDF Page | ~0.8-1.2s | Per page, includes OCR |
| Signature Detection | ~50-100ms | 3 parallel methods |
| OCR Extraction | ~200-300ms | Tesseract + TrOCR fallback |

**System Requirements:**
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor (PyTorch benefits from parallelization)
- **GPU**: Optional (speeds up YOLO and LayoutLM inference)

---

## 🔒 Security & Compliance

### **Audit Logging**
Every redaction operation generates:
- Unique audit ID (UUID)
- Timestamp (ISO 8601)
- Detection details (type, location, confidence)
- Risk assessment (HIGH/MEDIUM/LOW)
- Processing metadata

### **Data Privacy**
- ✅ Local processing (no cloud uploads)
- ✅ No data retention (except audit logs)
- ✅ Configurable redaction opacity
- ✅ GDPR/HIPAA awareness built-in

---

## 🐛 Troubleshooting

### **Common Issues**

1. **"Tesseract not found"**
   ```bash
   # Install Tesseract OCR
   # Windows: https://github.com/UB-Mannheim/tesseract/wiki
   # Linux: sudo apt install tesseract-ocr
   # Mac: brew install tesseract
   ```

2. **"Unable to get page count. Is poppler installed?"**
   ```bash
   # Download Poppler for Windows
   # Extract to project root as poppler-25.07.0/
   # Or add to system PATH
   ```

3. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

4. **Slow processing**
   - Reduce image resolution before processing
   - Use GPU if available (PyTorch CUDA)
   - Disable visual detection for faster results

---

## 🛠️ Advanced Usage

### **Batch Processing**

```python
from pathlib import Path
from src.redactor import PIIRedactor
import cv2

redactor = PIIRedactor()
input_dir = Path('input_images')
output_dir = Path('output_redacted')

for img_path in input_dir.glob('*.jpg'):
    image = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = redactor.redact_image(image_rgb)
    
    output_path = output_dir / f"{img_path.stem}_REDACTED.jpg"
    cv2.imwrite(str(output_path), 
                cv2.cvtColor(result['redacted_image'], cv2.COLOR_RGB2BGR))
    print(f"✅ {img_path.name}: {len(result['detections'])} detections")
```

### **Custom Detection Rules**

Edit `src/redactor.py` to add custom patterns:

```python
# Add custom regex pattern
CUSTOM_PATTERNS = {
    'PASSPORT': r'\b[A-Z]{1,2}\d{6,9}\b',
    'LICENSE': r'\b[A-Z]{2}\d{2}\s\d{11}\b'
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Privara Intellectus Team**
- GitHub: [@ssundxr](https://github.com/ssundxr)

---

## 🙏 Acknowledgments

- **Tesseract OCR** - Google's open-source OCR engine
- **Hugging Face** - Transformers library and pre-trained models
- **Ultralytics** - YOLO object detection
- **OpenCV** - Computer vision toolkit
- **PyTorch** - Deep learning framework

---

## 📞 Support

For issues, questions, or feature requests:
- 📧 Open an issue on GitHub
- 📖 Check the documentation in `/docs`
- 💬 Join our community discussions

---

## 🗺️ Roadmap

- [ ] GPU acceleration optimization
- [ ] Cloud deployment support (AWS/Azure/GCP)
- [ ] API endpoint for integration
- [ ] Mobile app (React Native)
- [ ] Additional language support (Spanish, French, German)
- [ ] Real-time video redaction
- [ ] Custom model training interface

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by Privara Intellectus Team

</div>
