"""
Privara PII Redactor - Premium Enterprise UI
Professional interface with modern design and smooth animations
Version: 3.0 Enterprise - Premium Edition
"""


import os
import sys
import json
import threading
from pathlib import Path
from queue import Queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Modern Color Palette - Premium Design
COLORS = {
    # Primary Colors
    'primary': '#6366f1',  # Indigo
    'primary_dark': '#4f46e5',
    'primary_light': '#818cf8',
    
    # Accent Colors
    'accent': '#8b5cf6',  # Purple
    'success': '#10b981',  # Green
    'warning': '#f59e0b',  # Amber
    'error': '#ef4444',  # Red
    'info': '#3b82f6',  # Blue
    
    # Neutral Colors
    'bg_primary': '#0f172a',  # Dark slate
    'bg_secondary': '#1e293b',
    'bg_tertiary': '#334155',
    'surface': '#1e293b',
    'surface_light': '#334155',
    
    # Text Colors
    'text_primary': '#f1f5f9',
    'text_secondary': '#cbd5e1',
    'text_muted': '#94a3b8',
    
    # Border Colors
    'border': '#334155',
    'border_light': '#475569',
    
    # Gradient Colors
    'gradient_start': '#6366f1',
    'gradient_end': '#8b5cf6',
}


# Add project to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))


# Import redactor
try:
    from redactor import get_enhanced_redactor
    REDACTOR = get_enhanced_redactor()
    logger.info("✓ Enhanced redactor initialized")
except ImportError as e:
    logger.error(f"Failed to import redactor: {e}")
    from redactor import PIIRedactor
    REDACTOR = PIIRedactor()
    logger.info("✓ Basic redactor initialized")

# Import PDF redactor
try:
    from pdf_redactor import PDFRedactor
    PDF_REDACTOR = PDFRedactor()
    PDF_AVAILABLE = True
    logger.info("✓ PDF redactor initialized")
except Exception as e:
    PDF_REDACTOR = None
    PDF_AVAILABLE = False
    logger.warning(f"PDF redactor unavailable: {e}")



class ModernButton(tk.Canvas):
    """Custom modern button with hover effects and animations."""
    
    def __init__(self, parent, text, command=None, bg=COLORS['primary'], 
                 fg=COLORS['text_primary'], width=200, height=48, icon=None, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_secondary'], highlightthickness=0, **kwargs)
        
        self.text = text
        self.command = command
        self.bg_color = bg
        self.fg_color = fg
        self.width = width
        self.height = height
        self.icon = icon
        self.is_hovered = False
        self.is_disabled = False
        
        self.draw_button()
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
    
    def draw_button(self):
        """Draw the button with rounded corners."""
        self.delete('all')
        
        # Determine color based on state
        if self.is_disabled:
            bg = COLORS['surface']
            fg = COLORS['text_muted']
        elif self.is_hovered:
            bg = self.bg_color
            fg = self.fg_color
        else:
            # Slightly darker when not hovered
            bg = self.bg_color
            fg = self.fg_color
        
        # Draw rounded rectangle
        self.create_rounded_rectangle(2, 2, self.width-2, self.height-2, 
                                     radius=8, fill=bg, outline='')
        
        # Draw text
        text_y = self.height // 2
        if self.icon:
            # Draw icon + text
            self.create_text(self.width // 2 - 30, text_y, text=self.icon, 
                           fill=fg, font=('Segoe UI', 14))
            self.create_text(self.width // 2 + 10, text_y, text=self.text,
                           fill=fg, font=('Segoe UI', 11, 'bold'))
        else:
            self.create_text(self.width // 2, text_y, text=self.text,
                           fill=fg, font=('Segoe UI', 11, 'bold'))
    
    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=10, **kwargs):
        """Create a rounded rectangle."""
        points = [
            x1+radius, y1,
            x1+radius, y1,
            x2-radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, event):
        """Handle mouse enter."""
        if not self.is_disabled:
            self.is_hovered = True
            self.draw_button()
            self.config(cursor='hand2')
    
    def on_leave(self, event):
        """Handle mouse leave."""
        self.is_hovered = False
        self.draw_button()
        self.config(cursor='')
    
    def on_click(self, event):
        """Handle button click."""
        if not self.is_disabled and self.command:
            self.command()
    
    def set_state(self, state):
        """Set button state (normal/disabled)."""
        self.is_disabled = (state == 'disabled')
        self.draw_button()


class ModernCard(tk.Frame):
    """Modern card component with shadow effect."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], **kwargs)
        self.config(relief=tk.FLAT, borderwidth=0)


class PIIRedactorUI(tk.Tk):
    """
    Professional UI for PII Redactor.
    
    Features:
    - Image preview
    - Real-time processing
    - Audit log viewer
    - NLP explanations
    - Risk assessment
    - Batch processing
    """
    
    def __init__(self):
        super().__init__()
        
        self.title("PII Redactor - Enterprise Edition v3.0 (PDF Support)")
        self.geometry("1400x900")
        self.configure(bg='#f0f0f0')
        
        # State variables
        self.current_image = None
        self.redacted_image = None
        self.audit_data = None
        self.processing = False
        self.current_pdf_path = None  # For PDF processing
        self.is_pdf_mode = False  # Track if processing PDF
        
        # Setup UI
        self._setup_ui()
        
        logger.info("UI initialized")
    
    def _setup_ui(self):
        """Setup main UI layout."""
        # Menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Open PDF", command=self.load_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Save Redacted", command=self.save_redacted)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="View Audit Logs", command=self.view_audit_logs)
        tools_menu.add_command(label="Batch Process", command=self.batch_process)
        
        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        
        # Logo/Title
        title_label = ttk.Label(
            left_panel,
            text="PII REDACTOR",
            font=("Helvetica", 24, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(
            left_panel,
            text="Multimodal PII Redaction",
            font=("Helvetica", 10),
            foreground="#7f8c8d"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Action buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=10)
        
        self.btn_load = ttk.Button(
            btn_frame,
            text="📁 Load Image",
            command=self.load_image,
            width=25
        )
        self.btn_load.pack(pady=5)
        
        self.btn_load_pdf = ttk.Button(
            btn_frame,
            text="📄 Load PDF",
            command=self.load_pdf,
            width=25
        )
        self.btn_load_pdf.pack(pady=5)
        
        self.btn_process = ttk.Button(
            btn_frame,
            text="🔒 Redact PII",
            command=self.process_image,
            state="disabled",
            width=25
        )
        self.btn_process.pack(pady=5)
        
        self.btn_save = ttk.Button(
            btn_frame,
            text="💾 Save Result",
            command=self.save_redacted,
            state="disabled",
            width=25
        )
        self.btn_save.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            left_panel,
            mode='indeterminate',
            length=280
        )
        self.progress.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(
            left_panel,
            text="Ready",
            font=("Helvetica", 9),
            foreground="#27ae60"
        )
        self.status_label.pack(pady=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.pack(fill="both", expand=True, pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            height=10,
            width=30,
            font=("Courier", 9),
            wrap=tk.WORD
        )
        self.stats_text.pack(fill="both", expand=True)
        
        # Middle panel - Image viewer
        middle_panel = ttk.Frame(main_frame)
        middle_panel.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_columnconfigure(1, weight=2)
        
        # Original image
        orig_label = ttk.Label(middle_panel, text="Original", font=("Helvetica", 12, "bold"))
        orig_label.grid(row=0, column=0, pady=5)
        
        self.original_canvas = tk.Canvas(
            middle_panel,
            width=400,
            height=600,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.original_canvas.grid(row=1, column=0, padx=5, sticky="nsew")
        
        # Redacted image
        redacted_label = ttk.Label(middle_panel, text="Redacted", font=("Helvetica", 12, "bold"))
        redacted_label.grid(row=0, column=1, pady=5)
        
        self.redacted_canvas = tk.Canvas(
            middle_panel,
            width=400,
            height=600,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.redacted_canvas.grid(row=1, column=1, padx=5, sticky="nsew")
        
        middle_panel.grid_rowconfigure(1, weight=1)
        middle_panel.grid_columnconfigure(0, weight=1)
        middle_panel.grid_columnconfigure(1, weight=1)
        
        # Right panel - Audit & NLP
        right_panel = ttk.Frame(main_frame, width=350)
        right_panel.grid(row=0, column=2, sticky="ns", padx=(10, 0))
        
        # NLP Explanation
        nlp_frame = ttk.LabelFrame(right_panel, text="NLP Explanation", padding="10")
        nlp_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.nlp_text = scrolledtext.ScrolledText(
            nlp_frame,
            height=10,
            width=40,
            font=("Helvetica", 10),
            wrap=tk.WORD
        )
        self.nlp_text.pack(fill="both", expand=True)
        
        # Risk Assessment
        risk_frame = ttk.LabelFrame(right_panel, text="Risk Assessment", padding="10")
        risk_frame.pack(fill="x", pady=(0, 10))
        
        self.risk_label = ttk.Label(
            risk_frame,
            text="No analysis yet",
            font=("Helvetica", 10),
            foreground="#7f8c8d",
            wraplength=320,
            justify="left"
        )
        self.risk_label.pack(fill="x")
        
        # Audit Trail
        audit_frame = ttk.LabelFrame(right_panel, text="Audit Trail", padding="10")
        audit_frame.pack(fill="both", expand=True)
        
        self.audit_text = scrolledtext.ScrolledText(
            audit_frame,
            height=15,
            width=40,
            font=("Courier", 8),
            wrap=tk.WORD
        )
        self.audit_text.pack(fill="both", expand=True)
    
    def load_image(self):
        """Load image from file."""
        filepath = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            self.current_image = Image.open(filepath)
            self.display_image(self.current_image, self.original_canvas)
            
            self.is_pdf_mode = False
            self.current_pdf_path = None
            self.btn_process.config(state="normal")
            self.status_label.config(text="Image loaded", foreground="#27ae60")
            
            logger.info(f"Loaded image: {filepath}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            logger.error(f"Image load failed: {e}")
    
    def load_pdf(self):
        """Load PDF file."""
        if not PDF_AVAILABLE:
            messagebox.showerror(
                "PDF Not Available",
                "PDF support not installed.\n\n"
                "Install with:\n"
                "pip install pdf2image img2pdf\n\n"
                "Also install poppler:\n"
                "Windows: Download from poppler.freedesktop.org\n"
                "Linux: sudo apt-get install poppler-utils\n"
                "Mac: brew install poppler"
            )
            return
        
        filepath = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Check file size (rough estimate of page count)
            import os
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            if file_size_mb > 50:  # Rough limit
                response = messagebox.askyesno(
                    "Large PDF",
                    f"PDF file is {file_size_mb:.1f}MB.\n"
                    f"This may take several minutes.\n\n"
                    f"Continue?"
                )
                if not response:
                    return
            
            self.current_pdf_path = filepath
            self.is_pdf_mode = True
            self.current_image = None
            
            # Show PDF info
            from pathlib import Path
            pdf_name = Path(filepath).name
            
            # Clear canvases
            self.original_canvas.delete("all")
            self.redacted_canvas.delete("all")
            
            # Show PDF icon/text
            self.original_canvas.create_text(
                200, 300,
                text=f"📄 PDF Loaded\n\n{pdf_name}\n\nClick 'Redact PII' to process",
                font=("Helvetica", 14),
                fill="#2c3e50"
            )
            
            self.btn_process.config(state="normal")
            self.status_label.config(text="PDF loaded", foreground="#27ae60")
            
            logger.info(f"Loaded PDF: {filepath}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF:\n{e}")
            logger.error(f"PDF load failed: {e}")
    
    def process_image(self):
        """Process image or PDF in background thread."""
        if self.processing:
            return
        
        if not self.current_image and not self.current_pdf_path:
            return
        
        self.processing = True
        self.btn_process.config(state="disabled")
        self.btn_load.config(state="disabled")
        self.btn_load_pdf.config(state="disabled")
        self.progress.start()
        
        if self.is_pdf_mode:
            self.status_label.config(text="Processing PDF...", foreground="#e67e22")
        else:
            self.status_label.config(text="Processing...", foreground="#e67e22")
        
        # Process in thread
        thread = threading.Thread(target=self._process_worker, daemon=True)
        thread.start()
    
    def _process_worker(self):
        """Background processing worker."""
        try:
            if self.is_pdf_mode:
                # Process PDF
                output_path, audit = PDF_REDACTOR.redact_pdf(
                    self.current_pdf_path,
                    progress_callback=self._pdf_progress_callback
                )
                
                # Update UI in main thread
                self.after(0, self._pdf_process_complete, output_path, audit)
            else:
                # Process image
                redacted, audit = REDACTOR.redact_image(
                    self.current_image,
                    filename="document.jpg",
                    generate_audit=True
                )
                
                # Update UI in main thread
                self.after(0, self._process_complete, redacted, audit)
        
        except Exception as e:
            self.after(0, self._process_error, str(e))
    
    def _pdf_progress_callback(self, page: int, total: int, status: str):
        """Callback for PDF processing progress."""
        def update_status():
            if total > 0:
                self.status_label.config(
                    text=f"Processing page {page}/{total}...",
                    foreground="#e67e22"
                )
        
        self.after(0, update_status)
    
    def _process_complete(self, redacted: Image.Image, audit: dict):
        """Handle processing completion."""
        self.redacted_image = redacted
        self.audit_data = audit
        
        # Display redacted image
        self.display_image(redacted, self.redacted_canvas)
        
        # Update statistics
        self.update_statistics(audit['statistics'])
        
        # Update NLP explanation
        self.nlp_text.delete(1.0, tk.END)
        self.nlp_text.insert(1.0, audit['nlp_explanation'])
        
        # Update risk assessment
        risk_text = audit['risk_assessment']
        risk_color = self._get_risk_color(risk_text)
        self.risk_label.config(text=risk_text, foreground=risk_color)
        
        # Update audit trail
        self.update_audit_trail(audit)
        
        # Re-enable buttons
        self.processing = False
        self.progress.stop()
        self.btn_process.config(state="normal")
        self.btn_load.config(state="normal")
        self.btn_load_pdf.config(state="normal")
        self.btn_save.config(state="normal")
        self.status_label.config(text="Complete", foreground="#27ae60")
        
        logger.info("Processing complete")
    
    def _pdf_process_complete(self, output_path: str, audit: dict):
        """Handle PDF processing completion."""
        self.audit_data = audit
        self.redacted_image = None  # PDF, not image
        
        # Show success message on canvas
        self.redacted_canvas.delete("all")
        self.redacted_canvas.create_text(
            200, 250,
            text=f"✅ PDF Redacted Successfully!\n\n"
                 f"Pages: {audit['total_pages']}\n"
                 f"Detections: {audit['total_detections']}\n"
                 f"Time: {audit['processing_time']}s\n\n"
                 f"Saved to:\n{Path(output_path).name}",
            font=("Helvetica", 12),
            fill="#27ae60",
            width=350
        )
        
        # Update statistics
        stats = audit['statistics']
        self.stats_text.delete(1.0, tk.END)
        text = f"PDF Statistics:\n\n"
        text += f"Total Pages: {stats['total_pages']}\n"
        text += f"Total Detections: {stats['total_detections']}\n"
        text += f"Pages with PII: {stats['pages_with_pii']}\n"
        text += f"Pages without PII: {stats['pages_without_pii']}\n\n"
        text += f"By Risk:\n"
        for risk, count in stats['by_risk'].items():
            text += f"  {risk}: {count}\n"
        text += f"\nBy Type:\n"
        for type_, count in stats['by_type'].items():
            text += f"  {type_}: {count}\n"
        
        if stats['high_risk_pages']:
            text += f"\nHigh-Risk Pages:\n"
            text += f"  {', '.join(map(str, stats['high_risk_pages']))}\n"
        
        self.stats_text.insert(1.0, text)
        
        # Update NLP explanation
        self.nlp_text.delete(1.0, tk.END)
        explanation = f"Processed {stats['total_pages']}-page PDF document. "
        explanation += f"Detected and redacted {stats['total_detections']} PII items across {stats['pages_with_pii']} pages. "
        explanation += f"Processing completed in {audit['processing_time']} seconds. "
        explanation += f"Redacted PDF is ready for download."
        self.nlp_text.insert(1.0, explanation)
        
        # Update risk assessment
        if stats['by_risk']['HIGH'] >= 10:
            risk_text = "CRITICAL - Multiple high-risk PII across document"
            risk_color = "#e74c3c"
        elif stats['by_risk']['HIGH'] >= 5:
            risk_text = "HIGH - Significant PII detected"
            risk_color = "#e67e22"
        elif stats['total_detections'] >= 10:
            risk_text = "MODERATE - Multiple PII items"
            risk_color = "#f39c12"
        else:
            risk_text = "LOW - Minimal PII detected"
            risk_color = "#27ae60"
        
        self.risk_label.config(text=risk_text, foreground=risk_color)
        
        # Update audit trail
        self.audit_text.delete(1.0, tk.END)
        text = f"PDF Audit Trail\n\n"
        text += f"Filename: {audit['filename']}\n"
        text += f"Timestamp: {audit['timestamp']}\n"
        text += f"Processing Time: {audit['processing_time']}s\n"
        text += f"DPI: {audit['dpi']}\n\n"
        text += f"--- Pages Summary ---\n"
        for page in audit['pages'][:10]:
            text += f"Page {page['page_number']}: {page['detections']} detections\n"
        
        if len(audit['pages']) > 10:
            text += f"\n... and {len(audit['pages']) - 10} more pages\n"
        
        self.audit_text.insert(1.0, text)
        
        # Re-enable buttons
        self.processing = False
        self.progress.stop()
        self.btn_process.config(state="normal")
        self.btn_load.config(state="normal")
        self.btn_load_pdf.config(state="normal")
        self.btn_save.config(state="normal")
        self.status_label.config(text="PDF Complete", foreground="#27ae60")
        
        # Show download option
        messagebox.showinfo(
            "PDF Redacted",
            f"PDF successfully redacted!\n\n"
            f"Output: {Path(output_path).name}\n"
            f"Location: {Path(output_path).parent}\n\n"
            f"Click 'Save Result' to choose a different location."
        )
        
        # Store output path for saving
        self.current_pdf_output = output_path
        
        logger.info(f"PDF processing complete: {output_path}")
    
    def _process_error(self, error_msg: str):
        """Handle processing error."""
        self.processing = False
        self.progress.stop()
        self.btn_process.config(state="normal")
        self.btn_load.config(state="normal")
        self.btn_load_pdf.config(state="normal")
        self.status_label.config(text="Error", foreground="#e74c3c")
        
        messagebox.showerror("Processing Error", f"Failed to process:\n{error_msg}")
        logger.error(f"Processing failed: {error_msg}")
    
    def display_image(self, image: Image.Image, canvas: tk.Canvas):
        """Display image on canvas with aspect ratio."""
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 400
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 600
        
        # Resize to fit
        img_copy = image.copy()
        img_copy.thumbnail((canvas_width - 20, canvas_height - 20), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img_copy)
        
        # Clear canvas
        canvas.delete("all")
        
        # Center image
        x = (canvas_width - img_copy.width) // 2
        y = (canvas_height - img_copy.height) // 2
        
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep reference
    
    def update_statistics(self, stats: dict):
        """Update statistics display."""
        self.stats_text.delete(1.0, tk.END)
        
        text = f"Total Detections: {stats['total_detections']}\n\n"
        
        text += "By Risk:\n"
        for risk, count in stats['by_risk'].items():
            text += f"  {risk}: {count}\n"
        
        text += "\nBy Type:\n"
        for type_, count in stats['by_type'].items():
            text += f"  {type_}: {count}\n"
        
        text += "\nTop Entities:\n"
        for entity, count in list(stats['entities'].items())[:5]:
            text += f"  {entity}: {count}\n"
        
        self.stats_text.insert(1.0, text)
    
    def update_audit_trail(self, audit: dict):
        """Update audit trail display."""
        self.audit_text.delete(1.0, tk.END)
        
        text = f"Timestamp: {audit['timestamp']}\n"
        text += f"Filename: {audit['filename']}\n"
        text += f"Processing Time: {audit['processing_time']}s\n"
        text += f"Image Size: {audit['image_size']}\n\n"
        
        text += "--- Detections ---\n"
        for i, det in enumerate(audit['detections'][:10], 1):
            text += f"{i}. {det['entity']} ({det['risk']})\n"
            text += f"   Source: {det.get('source', 'N/A')}\n"
            text += f"   Confidence: {det['confidence']:.2f}\n\n"
        
        if len(audit['detections']) > 10:
            text += f"... and {len(audit['detections']) - 10} more\n"
        
        self.audit_text.insert(1.0, text)
    
    def _get_risk_color(self, risk_text: str) -> str:
        """Get color for risk level."""
        if "CRITICAL" in risk_text:
            return "#e74c3c"
        elif "HIGH" in risk_text:
            return "#e67e22"
        elif "MODERATE" in risk_text:
            return "#f39c12"
        else:
            return "#27ae60"
    
    def save_redacted(self):
        """Save redacted image or PDF."""
        # Check if PDF mode
        if self.is_pdf_mode and hasattr(self, 'current_pdf_output'):
            # Copy PDF to chosen location
            import shutil
            
            filepath = filedialog.asksaveasfilename(
                title="Save Redacted PDF",
                defaultextension=".pdf",
                filetypes=[
                    ("PDF", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if filepath:
                try:
                    shutil.copy2(self.current_pdf_output, filepath)
                    messagebox.showinfo("Saved", f"PDF saved to:\n{filepath}")
                    logger.info(f"Saved redacted PDF: {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save PDF:\n{e}")
            return
        
        # Image mode
        if not self.redacted_image:
            messagebox.showwarning("No Image", "No redacted image to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Redacted Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                self.redacted_image.save(filepath)
                messagebox.showinfo("Saved", f"Image saved to:\n{filepath}")
                logger.info(f"Saved redacted image: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}")
    
    def view_audit_logs(self):
        """Open audit logs directory."""
        audit_dir = Path("output/audit_logs")
        if audit_dir.exists():
            os.startfile(audit_dir) if os.name == 'nt' else os.system(f'open "{audit_dir}"')
        else:
            messagebox.showinfo("No Logs", "No audit logs found yet")
    
    def batch_process(self):
        """Batch process multiple images."""
        messagebox.showinfo("Coming Soon", "Batch processing will be available in the next update")



def main():
    """Launch PII Redactor UI."""
    app = PIIRedactorUI()
    app.mainloop()



if __name__ == "__main__":
    main()
