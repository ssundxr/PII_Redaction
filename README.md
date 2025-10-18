# PII Redaction System

A modular Python project for detecting and redacting Personally Identifiable Information (PII) from images and text. It provides OCR to extract text from images, regex-based PII detection, and utilities to redact detected spans.

## Features

- OCR via `pytesseract` to extract text and word bounding boxes.
- Simple regex-based PII detectors (e.g., emails, phone numbers).
- Text redaction and image box redaction utilities.
- Extensible structure for plugging in advanced AI models under `models/`.

## Project Structure

```
pii-redaction-system/
├─ src/
│  ├─ __init__.py
│  ├─ ocr.py
│  ├─ detector.py
│  ├─ redactor.py
│  └─ ui.py
├─ models/
├─ tests/
│  └─ sample_images/
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## Setup

- Python 3.9+
- Tesseract OCR installed and available on PATH (or set the `pytesseract.pytesseract.tesseract_cmd`).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simple CLI that performs OCR on an image, detects PII in the text, and prints the redacted text:

```bash
python -m src.ui path/to/image.png
```

## Notes

- Place large or custom AI models under `models/` (ignored by git by default).
- The detectors in `src/detector.py` are extensible. You can add patterns or integrate ML models as needed.
