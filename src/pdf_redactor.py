"""
PDF Redactor - Multi-page PDF PII Redaction
Converts PDF to images, redacts PII, converts back to PDF
Version: 1.0 Enterprise
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Try importing PDF libraries
try:
    from pdf2image import convert_from_path
    import pdf2image.exceptions
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available - install: pip install pdf2image")

try:
    from PIL import Image
    import img2pdf
    IMG2PDF_AVAILABLE = True
except ImportError:
    IMG2PDF_AVAILABLE = False
    logger.warning("img2pdf not available - install: pip install img2pdf")

# Import redactor
from redactor import PIIRedactor


class PDFRedactor:
    """
    Multi-page PDF PII Redactor.
    
    Features:
    - Converts PDF to images (page by page)
    - Redacts PII on each page
    - Converts back to PDF
    - Supports up to 100 pages
    - Generates comprehensive audit log
    """
    
    MAX_PAGES = 100
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize PDF redactor.
        
        Args:
            output_dir: Output directory for redacted PDFs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pdf_output_dir = self.output_dir / "pdf_redacted"
        self.pdf_output_dir.mkdir(exist_ok=True)
        
        self.pdf_audit_dir = self.output_dir / "pdf_audit_logs"
        self.pdf_audit_dir.mkdir(exist_ok=True)
        
        # Initialize image redactor
        self.redactor = PIIRedactor(output_dir=str(output_dir))
        
        logger.info("✓ PDF Redactor initialized")
    
    def redact_pdf(
        self,
        pdf_path: str,
        output_filename: str = None,
        dpi: int = 200,
        progress_callback = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Redact PII from multi-page PDF.
        
        Args:
            pdf_path: Path to input PDF
            output_filename: Name for output PDF (auto-generated if None)
            dpi: DPI for PDF to image conversion (200 recommended)
            progress_callback: Optional callback(page, total_pages, status)
            
        Returns:
            Tuple of (output_pdf_path, audit_data)
        """
        if not PDF2IMAGE_AVAILABLE:
            raise Exception("pdf2image not installed. Run: pip install pdf2image")
        
        if not IMG2PDF_AVAILABLE:
            raise Exception("img2pdf not installed. Run: pip install img2pdf")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING PDF REDACTION: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Initialize audit data
        audit_data = {
            'filename': pdf_path.name,
            'timestamp': start_time.isoformat(),
            'dpi': dpi,
            'pages': [],
            'total_pages': 0,
            'total_detections': 0,
            'processing_time': 0,
            'statistics': {}
        }
        
        try:
            # Step 1: Convert PDF to images
            if progress_callback:
                progress_callback(0, 0, "Converting PDF to images...")
            
            logger.info("\n[1/3] Converting PDF to images...")
            images = self._convert_pdf_to_images(pdf_path, dpi)
            
            total_pages = len(images)
            audit_data['total_pages'] = total_pages
            
            if total_pages == 0:
                raise Exception("No pages found in PDF")
            
            if total_pages > self.MAX_PAGES:
                raise Exception(f"PDF has {total_pages} pages. Maximum allowed: {self.MAX_PAGES}")
            
            logger.info(f"✓ Converted {total_pages} pages")
            
            # Step 2: Redact each page
            logger.info(f"\n[2/3] Redacting {total_pages} pages...")
            redacted_images = []
            
            for page_num, image in enumerate(images, 1):
                if progress_callback:
                    progress_callback(page_num, total_pages, f"Redacting page {page_num}/{total_pages}")
                
                logger.info(f"\n--- Page {page_num}/{total_pages} ---")
                
                # Redact page
                redacted_img, page_audit = self.redactor.redact_image(
                    image,
                    filename=f"{pdf_path.stem}_page_{page_num}.jpg",
                    generate_audit=False
                )
                
                redacted_images.append(redacted_img)
                
                # Store page audit
                page_summary = {
                    'page_number': page_num,
                    'detections': len(page_audit['detections']),
                    'statistics': page_audit['statistics'],
                    'risk_assessment': page_audit.get('risk_assessment', 'N/A')
                }
                audit_data['pages'].append(page_summary)
                audit_data['total_detections'] += len(page_audit['detections'])
                
                logger.info(f"✓ Page {page_num}: {len(page_audit['detections'])} detections")
            
            # Step 3: Convert back to PDF
            if progress_callback:
                progress_callback(total_pages, total_pages, "Converting to PDF...")
            
            logger.info(f"\n[3/3] Converting to PDF...")
            
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{pdf_path.stem}_redacted_{timestamp}.pdf"
            
            output_path = self.pdf_output_dir / output_filename
            
            self._convert_images_to_pdf(redacted_images, output_path)
            
            logger.info(f"✓ Redacted PDF saved: {output_path.name}")
            
            # Finalize audit
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            audit_data['processing_time'] = round(processing_time, 2)
            audit_data['output_file'] = output_filename
            audit_data['statistics'] = self._calculate_pdf_statistics(audit_data['pages'])
            
            # Save audit
            self._save_audit(audit_data)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ PDF REDACTION COMPLETE")
            logger.info(f"  Total Pages: {total_pages}")
            logger.info(f"  Total Detections: {audit_data['total_detections']}")
            logger.info(f"  Processing Time: {processing_time:.2f}s")
            logger.info(f"  Output: {output_path}")
            logger.info(f"{'='*60}\n")
            
            return str(output_path), audit_data
        
        except Exception as e:
            logger.error(f"PDF redaction failed: {e}")
            raise
    
    def _convert_pdf_to_images(self, pdf_path: Path, dpi: int) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Images (one per page)
        """
        try:
            # Try to find poppler in common locations
            import os
            import shutil
            
            poppler_path = None
            
            # Check if poppler is in PATH
            if shutil.which("pdftoppm") is not None:
                poppler_path = None  # Use system PATH
            else:
                # Check common Windows locations
                common_paths = [
                    r"C:\Program Files\poppler\Library\bin",
                    r"C:\Program Files\poppler-24.08.0\Library\bin",
                    r"C:\Program Files (x86)\poppler\Library\bin",
                    r"C:\poppler\Library\bin",
                    os.path.join(os.getcwd(), "poppler", "Library", "bin"),
                    os.path.join(os.getcwd(), "poppler-25.07.0", "Library", "bin"),
                ]
                
                for path in common_paths:
                    if os.path.exists(path) and os.path.exists(os.path.join(path, "pdftoppm.exe")):
                        poppler_path = path
                        logger.info(f"Found poppler at: {poppler_path}")
                        break
            
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt='png',
                thread_count=4,  # Parallel processing
                grayscale=False,
                poppler_path=poppler_path
            )
            return images
        except pdf2image.exceptions.PDFPageCountError as e:
            logger.error(f"Invalid PDF: {e}")
            raise Exception("Invalid or corrupted PDF file")
        except pdf2image.exceptions.PDFInfoNotInstalledError as e:
            logger.error(f"Poppler not found: {e}")
            raise Exception(
                "Poppler not found! Please install:\n"
                "1. Download from: https://github.com/oschwartz10612/poppler-windows/releases\n"
                "2. Extract to C:\\Program Files\\poppler\n"
                "3. Add to PATH: C:\\Program Files\\poppler\\Library\\bin\n"
                "OR place poppler folder in project directory"
            )
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise Exception(f"Failed to convert PDF: {e}")
    
    def _convert_images_to_pdf(self, images: List[Image.Image], output_path: Path):
        """
        Convert list of images to PDF.
        
        Args:
            images: List of PIL Images
            output_path: Output PDF path
        """
        try:
            # Convert images to bytes
            image_bytes = []
            for img in images:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                from io import BytesIO
                img_buffer = BytesIO()
                img.save(img_buffer, format='JPEG', quality=95)
                image_bytes.append(img_buffer.getvalue())
            
            # Create PDF
            with open(output_path, 'wb') as f:
                f.write(img2pdf.convert(image_bytes))
            
            logger.info(f"✓ PDF created: {output_path.name} ({len(images)} pages)")
        
        except Exception as e:
            logger.error(f"PDF creation failed: {e}")
            raise Exception(f"Failed to create PDF: {e}")
    
    def _calculate_pdf_statistics(self, pages: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics for PDF."""
        stats = {
            'total_pages': len(pages),
            'total_detections': sum(p['detections'] for p in pages),
            'pages_with_pii': sum(1 for p in pages if p['detections'] > 0),
            'pages_without_pii': sum(1 for p in pages if p['detections'] == 0),
            'avg_detections_per_page': 0,
            'by_risk': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'by_type': {},
            'high_risk_pages': []
        }
        
        if len(pages) > 0:
            stats['avg_detections_per_page'] = round(
                stats['total_detections'] / len(pages), 2
            )
        
        # Aggregate statistics
        for page in pages:
            page_stats = page.get('statistics', {})
            
            # By risk
            for risk, count in page_stats.get('by_risk', {}).items():
                if risk in stats['by_risk']:
                    stats['by_risk'][risk] += count
            
            # By type
            for type_, count in page_stats.get('by_type', {}).items():
                stats['by_type'][type_] = stats['by_type'].get(type_, 0) + count
            
            # Track high-risk pages
            if page_stats.get('by_risk', {}).get('HIGH', 0) >= 3:
                stats['high_risk_pages'].append(page['page_number'])
        
        return stats
    
    def _save_audit(self, audit_data: Dict):
        """Save PDF audit log."""
        try:
            audit_id = hashlib.sha256(
                f"{audit_data['filename']}{audit_data['timestamp']}".encode()
            ).hexdigest()[:12]
            
            audit_file = self.pdf_audit_dir / f"pdf_audit_{audit_id}.json"
            
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ PDF audit saved: {audit_file.name}")
        except Exception as e:
            logger.error(f"Audit save failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get PDF redactor information."""
        return {
            'max_pages': self.MAX_PAGES,
            'pdf2image_available': PDF2IMAGE_AVAILABLE,
            'img2pdf_available': IMG2PDF_AVAILABLE,
            'output_directory': str(self.pdf_output_dir),
            'audit_directory': str(self.pdf_audit_dir)
        }


def redact_pdf_simple(pdf_path: str, output_filename: str = None) -> str:
    """
    Simple function to redact a PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_filename: Optional output filename
        
    Returns:
        Path to redacted PDF
    """
    redactor = PDFRedactor()
    output_path, _ = redactor.redact_pdf(pdf_path, output_filename)
    return output_path
