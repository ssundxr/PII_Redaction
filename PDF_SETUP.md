# PDF Support Installation Guide

## Required Dependencies

### 1. Python Packages
```bash
pip install pdf2image img2pdf
```

### 2. Poppler (Required for pdf2image)

Poppler is needed to convert PDFs to images.

#### **Windows:**
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler`
3. Add to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" variable
   - Add: `C:\Program Files\poppler\Library\bin`
4. Restart terminal/IDE

**OR** use Conda:
```bash
conda install -c conda-forge poppler
```

#### **Linux:**
```bash
sudo apt-get install poppler-utils
```

#### **Mac:**
```bash
brew install poppler
```

---

## Quick Install (All-in-one)

### Windows (PowerShell as Administrator):
```powershell
# Install Python packages
pip install pdf2image img2pdf

# Download and setup poppler
$url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip"
$output = "$env:TEMP\poppler.zip"
Invoke-WebRequest -Uri $url -OutFile $output

Expand-Archive -Path $output -DestinationPath "C:\Program Files\poppler" -Force

# Add to PATH
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\poppler\Library\bin", "Machine")

Write-Host "✓ Poppler installed. Restart your terminal."
```

### Linux:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
pip install pdf2image img2pdf
```

### Mac:
```bash
brew install poppler
pip install pdf2image img2pdf
```

---

## Verify Installation

Run this to verify:
```bash
python -c "from pdf2image import convert_from_path; print('✓ PDF support ready')"
```

If you see `✓ PDF support ready`, you're all set!

---

## Test PDF Redaction

```bash
python test_pdf_redaction.py
```

This will:
1. Create a sample 3-page PDF with PII
2. Redact all PII from each page
3. Save the redacted PDF
4. Generate audit logs

---

## Features

✅ **Multi-page Support** - Up to 100 pages per PDF
✅ **Automatic Page Processing** - Each page redacted individually
✅ **Comprehensive Audit** - Page-by-page PII statistics
✅ **Progress Tracking** - Real-time status updates
✅ **Download Ready** - Redacted PDF ready for download

---

## Usage in GUI

1. Launch: `python -m src.ui`
2. Click **"📄 Load PDF"**
3. Select your PDF (up to 100 pages)
4. Click **"🔒 Redact PII"**
5. Wait for processing (shows progress per page)
6. Click **"💾 Save Result"** to download

---

## Troubleshooting

### Error: "Unable to get page count"
- **Solution**: Poppler not installed or not in PATH
- **Fix**: Follow installation steps above

### Error: "pdf2image not found"
- **Solution**: `pip install pdf2image`

### Error: "img2pdf not found"
- **Solution**: `pip install img2pdf`

### Slow processing
- **Solution**: Reduce DPI (default: 200)
- **Fix**: Edit `src/pdf_redactor.py` line 67, change `dpi: int = 200` to `dpi: int = 150`

### Memory issues with large PDFs
- **Solution**: Process in smaller batches
- **Note**: Current limit is 100 pages

---

## Performance

| Pages | DPI | Time (approx) | File Size |
|-------|-----|---------------|-----------|
| 1-5   | 200 | 5-10s        | 500KB-2MB |
| 10-20 | 200 | 15-30s       | 2-5MB     |
| 50+   | 200 | 1-3 min      | 10-20MB   |
| 100   | 150 | 3-5 min      | 15-25MB   |

*Times are approximate and depend on system specs*

---

## Output

Redacted PDFs are saved to:
```
output/
├── pdf_redacted/
│   └── your_file_redacted_TIMESTAMP.pdf
└── pdf_audit_logs/
    └── pdf_audit_HASH.json
```

---

## Support

If you encounter issues:
1. Run: `python test_pdf_redaction.py`
2. Check error messages
3. Verify poppler installation: `poppler-utils --version` (Linux/Mac) or check PATH (Windows)
