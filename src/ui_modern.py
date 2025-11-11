"""
Modern Professional UI for Privara Intellectus PII Redaction System
Enterprise-grade design with smooth animations and intuitive UX
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

from redactor import PIIRedactor
from pdf_redactor import PDFRedactor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modern Color Palette - Enterprise Dark Theme
COLORS = {
    'bg_primary': '#0a0e27',      # Deep navy background
    'bg_secondary': '#141b2d',     # Secondary dark blue
    'surface': '#1e2746',          # Card/surface color
    'primary': '#6366f1',          # Indigo primary
    'accent': '#ec4899',           # Pink accent
    'success': '#10b981',          # Green success
    'warning': '#f59e0b',          # Amber warning
    'error': '#ef4444',            # Red error
    'text_primary': '#f8fafc',     # White text
    'text_secondary': '#cbd5e1',   # Gray text
    'text_muted': '#94a3b8',       # Muted gray
    'border': '#334155',           # Border color
    'hover': '#2563eb',            # Hover blue
}

# Typography
FONTS = {
    'header': ('Segoe UI', 24, 'bold'),
    'subheader': ('Segoe UI', 16, 'bold'),
    'body': ('Segoe UI', 11),
    'body_bold': ('Segoe UI', 11, 'bold'),
    'caption': ('Segoe UI', 9),
    'button': ('Segoe UI', 10, 'bold'),
}


class ModernButton(tk.Canvas):
    """Custom modern button with hover effects and smooth animations."""
    
    def __init__(self, parent, text, command=None, bg=None, fg=None, 
                 width=180, height=48, icon='', **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['surface'], highlightthickness=0, **kwargs)
        
        self.text = text
        self.command = command
        self.bg_color = bg or COLORS['primary']
        self.fg_color = fg or COLORS['text_primary']
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
        
        # Determine colors based on state
        if self.is_disabled:
            bg = COLORS['border']
            fg = COLORS['text_muted']
        elif self.is_hovered:
            # Lighter shade on hover
            bg = self.brighten_color(self.bg_color)
            fg = self.fg_color
        else:
            bg = self.bg_color
            fg = self.fg_color
        
        # Draw rounded rectangle background
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=10, fill=bg)
        
        # Draw text with icon
        text_y = self.height // 2
        if self.icon:
            full_text = f"{self.icon}  {self.text}"
        else:
            full_text = self.text
            
        self.create_text(self.width // 2, text_y, text=full_text,
                        fill=fg, font=FONTS['button'])
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        """Create a rounded rectangle on canvas."""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def brighten_color(self, hex_color):
        """Brighten a hex color by 15%."""
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Brighten
        r = min(255, int(r * 1.15))
        g = min(255, int(g * 1.15))
        b = min(255, int(b * 1.15))
        return f'#{r:02x}{g:02x}{b:02x}'
    
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
            # Visual feedback
            self.after(100, self.command)
    
    def set_state(self, state):
        """Set button state (normal/disabled)."""
        self.is_disabled = (state == 'disabled')
        self.draw_button()


class ModernCard(tk.Frame):
    """Modern card component with elevated look."""
    
    def __init__(self, parent, padding=20, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], **kwargs)
        self.config(relief=tk.FLAT, borderwidth=0, 
                   highlightbackground=COLORS['border'],
                   highlightthickness=1)
        
        # Inner padding frame
        self.inner_frame = tk.Frame(self, bg=COLORS['surface'])
        self.inner_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)


class StatsCard(tk.Canvas):
    """Statistics card with icon and numbers."""
    
    def __init__(self, parent, title, value, icon, color, width=220, height=100):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['surface'], highlightthickness=0)
        
        self.title = title
        self.value = value
        self.icon = icon
        self.color = color
        
        self.draw()
    
    def draw(self):
        """Draw the stats card."""
        # Background
        self.create_rounded_rect(0, 0, 220, 100, radius=12, 
                                fill=COLORS['surface'], outline=COLORS['border'])
        
        # Icon circle
        self.create_oval(20, 20, 60, 60, fill=self.color, outline='')
        self.create_text(40, 40, text=self.icon, font=('Segoe UI', 18), 
                        fill='white')
        
        # Value
        self.create_text(80, 35, text=str(self.value), anchor='w',
                        font=('Segoe UI', 24, 'bold'), fill=COLORS['text_primary'])
        
        # Title
        self.create_text(80, 60, text=self.title, anchor='w',
                        font=('Segoe UI', 10), fill=COLORS['text_muted'])
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        """Create rounded rectangle."""
        points = [x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
                 x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
                 x1, y2, x1, y2-radius, x1, y1+radius, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def update_value(self, new_value):
        """Update the displayed value."""
        self.value = new_value
        self.delete('all')
        self.draw()


class ProgressIndicator(tk.Canvas):
    """Circular progress indicator with percentage."""
    
    def __init__(self, parent, size=80):
        super().__init__(parent, width=size, height=size, 
                        bg=COLORS['surface'], highlightthickness=0)
        self.size = size
        self.progress = 0
        self.draw()
    
    def draw(self):
        """Draw the progress circle."""
        self.delete('all')
        center = self.size // 2
        radius = (self.size - 10) // 2
        
        # Background circle
        self.create_oval(5, 5, self.size-5, self.size-5, 
                        outline=COLORS['border'], width=3, fill=COLORS['surface'])
        
        # Progress arc
        if self.progress > 0:
            extent = int(360 * (self.progress / 100))
            self.create_arc(5, 5, self.size-5, self.size-5,
                          start=90, extent=-extent, 
                          outline=COLORS['primary'], width=3, style='arc')
        
        # Percentage text
        self.create_text(center, center, text=f"{self.progress}%",
                        font=('Segoe UI', 14, 'bold'), fill=COLORS['text_primary'])
    
    def set_progress(self, value):
        """Update progress (0-100)."""
        self.progress = max(0, min(100, value))
        self.draw()


class PrivaraModernUI(tk.Tk):
    """
    Modern Professional UI for Privara Intellectus PII Redaction System.
    
    Features:
    - Drag & drop file upload
    - Real-time processing visualization
    - Interactive statistics dashboard
    - Smooth animations and transitions
    - Professional enterprise design
    """
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Privara Intellectus - PII Redaction Platform")
        self.geometry("1600x1000")
        self.configure(bg=COLORS['bg_primary'])
        self.minsize(1400, 900)
        
        # Initialize engines
        self.redactor = PIIRedactor()
        self.pdf_redactor = PDFRedactor()
        
        # State variables
        self.current_image = None
        self.redacted_image = None
        self.audit_data = None
        self.processing = False
        self.current_file_path = None
        self.is_pdf_mode = False
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'processing_time': 0
        }
        
        # Create UI
        self.create_ui()
        
        logger.info("Privara Modern UI initialized")
    
    def create_ui(self):
        """Build the complete UI layout."""
        # ===== HEADER =====
        self.create_header()
        
        # ===== MAIN CONTENT =====
        content_frame = tk.Frame(self, bg=COLORS['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Upload and Controls
        left_panel = tk.Frame(content_frame, bg=COLORS['bg_primary'], width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        self.create_upload_section(left_panel)
        self.create_stats_section(left_panel)
        self.create_details_section(left_panel)
        
        # Right panel - Image Display
        right_panel = tk.Frame(content_frame, bg=COLORS['bg_primary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_image_display_section(right_panel)
        
        # ===== STATUS BAR =====
        self.create_status_bar()
    
    def create_header(self):
        """Create the modern header with gradient background."""
        header = tk.Frame(self, bg=COLORS['bg_secondary'], height=90)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Left side - Logo and Title
        left_container = tk.Frame(header, bg=COLORS['bg_secondary'])
        left_container.pack(side=tk.LEFT, padx=30, pady=20)
        
        # Logo
        logo_canvas = tk.Canvas(left_container, width=50, height=50,
                               bg=COLORS['bg_secondary'], highlightthickness=0)
        logo_canvas.pack(side=tk.LEFT, padx=(0, 15))
        # Draw shield logo
        logo_canvas.create_oval(5, 5, 45, 45, fill=COLORS['primary'], outline='')
        logo_canvas.create_text(25, 25, text='🛡️', font=('Segoe UI', 22))
        
        # Title
        title_frame = tk.Frame(left_container, bg=COLORS['bg_secondary'])
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(title_frame, text="Privara Intellectus",
                font=FONTS['header'], bg=COLORS['bg_secondary'],
                fg=COLORS['text_primary']).pack(anchor='w')
        tk.Label(title_frame, text="AI-Powered PII Redaction & Privacy Protection",
                font=FONTS['caption'], bg=COLORS['bg_secondary'],
                fg=COLORS['text_muted']).pack(anchor='w')
        
        # Right side - Quick Actions
        right_container = tk.Frame(header, bg=COLORS['bg_secondary'])
        right_container.pack(side=tk.RIGHT, padx=30, pady=20)
        
        ModernButton(right_container, "View Audit Logs", 
                    command=self.view_audit_logs,
                    bg=COLORS['border'], width=160, height=44).pack(side=tk.LEFT, padx=5)
        
        ModernButton(right_container, "Settings", 
                    command=self.open_settings,
                    bg=COLORS['border'], width=140, height=44).pack(side=tk.LEFT, padx=5)
    
    def create_upload_section(self, parent):
        """Create the file upload section."""
        card = ModernCard(parent)
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Title
        tk.Label(card.inner_frame, text="Upload Document",
                font=FONTS['subheader'], bg=COLORS['surface'],
                fg=COLORS['text_primary']).pack(anchor='w', pady=(0, 15))
        
        # Drag & drop zone (visual representation)
        upload_zone = tk.Frame(card.inner_frame, bg=COLORS['bg_secondary'],
                              height=180, highlightbackground=COLORS['border'],
                              highlightthickness=2)
        upload_zone.pack(fill=tk.X, pady=(0, 15))
        upload_zone.pack_propagate(False)
        
        # Upload icon and text
        tk.Label(upload_zone, text="📁", font=('Segoe UI', 48),
                bg=COLORS['bg_secondary']).pack(pady=(30, 10))
        tk.Label(upload_zone, text="Click to upload or drag & drop",
                font=FONTS['body_bold'], bg=COLORS['bg_secondary'],
                fg=COLORS['text_secondary']).pack()
        tk.Label(upload_zone, text="Supports: JPG, PNG, PDF (up to 100 pages)",
                font=FONTS['caption'], bg=COLORS['bg_secondary'],
                fg=COLORS['text_muted']).pack()
        
        # Bind click event
        upload_zone.bind('<Button-1>', lambda e: self.select_file())
        
        # Action buttons
        btn_frame = tk.Frame(card.inner_frame, bg=COLORS['surface'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.load_img_btn = ModernButton(btn_frame, "Load Image", 
                                         command=self.load_image,
                                         icon='🖼️', bg=COLORS['primary'],
                                         width=175, height=44)
        self.load_img_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.load_pdf_btn = ModernButton(btn_frame, "Load PDF", 
                                         command=self.load_pdf,
                                         icon='📄', bg=COLORS['primary'],
                                         width=175, height=44)
        self.load_pdf_btn.pack(side=tk.LEFT)
        
        # Process button
        self.redact_btn = ModernButton(card.inner_frame, "🔒 Redact PII", 
                                       command=self.redact_pii,
                                       bg=COLORS['accent'], width=360, height=50)
        self.redact_btn.pack(fill=tk.X, pady=(15, 0))
        self.redact_btn.set_state('disabled')
    
    def create_stats_section(self, parent):
        """Create statistics dashboard."""
        card = ModernCard(parent)
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Title
        tk.Label(card.inner_frame, text="Detection Statistics",
                font=FONTS['subheader'], bg=COLORS['surface'],
                fg=COLORS['text_primary']).pack(anchor='w', pady=(0, 15))
        
        # Stats grid
        stats_grid = tk.Frame(card.inner_frame, bg=COLORS['surface'])
        stats_grid.pack(fill=tk.X)
        
        # Row 1
        row1 = tk.Frame(stats_grid, bg=COLORS['surface'])
        row1.pack(fill=tk.X, pady=(0, 10))
        
        self.total_stat = StatsCard(row1, "Total PII", 0, "📊", COLORS['primary'])
        self.total_stat.pack(side=tk.LEFT, padx=(0, 10))
        
        self.time_stat = StatsCard(row1, "Time (sec)", "0.00", "⚡", COLORS['warning'])
        self.time_stat.pack(side=tk.LEFT)
        
        # Row 2
        row2 = tk.Frame(stats_grid, bg=COLORS['surface'])
        row2.pack(fill=tk.X)
        
        self.high_risk_stat = StatsCard(row2, "High Risk", 0, "🔴", COLORS['error'])
        self.high_risk_stat.pack(side=tk.LEFT, padx=(0, 10))
        
        self.medium_risk_stat = StatsCard(row2, "Medium Risk", 0, "🟡", COLORS['warning'])
        self.medium_risk_stat.pack(side=tk.LEFT)
    
    def create_details_section(self, parent):
        """Create details and actions section."""
        card = ModernCard(parent)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Title
        tk.Label(card.inner_frame, text="Actions & Details",
                font=FONTS['subheader'], bg=COLORS['surface'],
                fg=COLORS['text_primary']).pack(anchor='w', pady=(0, 15))
        
        # File info
        self.file_label = tk.Label(card.inner_frame, text="No file loaded",
                                   font=FONTS['body'], bg=COLORS['surface'],
                                   fg=COLORS['text_muted'], wraplength=320,
                                   justify=tk.LEFT)
        self.file_label.pack(anchor='w', pady=(0, 20))
        
        # Action buttons
        self.save_btn = ModernButton(card.inner_frame, "💾 Save Redacted",
                                     command=self.save_result,
                                     bg=COLORS['success'], width=360, height=48)
        self.save_btn.pack(fill=tk.X, pady=(0, 10))
        self.save_btn.set_state('disabled')
        
        self.export_audit_btn = ModernButton(card.inner_frame, "📋 Export Audit Log",
                                            command=self.export_audit,
                                            bg=COLORS['border'], width=360, height=48)
        self.export_audit_btn.pack(fill=tk.X, pady=(0, 10))
        self.export_audit_btn.set_state('disabled')
        
        self.clear_btn = ModernButton(card.inner_frame, "🗑️ Clear All",
                                     command=self.clear_all,
                                     bg=COLORS['error'], width=360, height=48)
        self.clear_btn.pack(fill=tk.X)
    
    def create_image_display_section(self, parent):
        """Create the image display area with before/after comparison."""
        # Title bar
        title_bar = tk.Frame(parent, bg=COLORS['bg_primary'])
        title_bar.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(title_bar, text="Document Preview",
                font=FONTS['subheader'], bg=COLORS['bg_primary'],
                fg=COLORS['text_primary']).pack(side=tk.LEFT)
        
        # Tab selector
        self.view_mode = tk.StringVar(value='original')
        tab_frame = tk.Frame(title_bar, bg=COLORS['surface'],
                            highlightbackground=COLORS['border'], highlightthickness=1)
        tab_frame.pack(side=tk.RIGHT)
        
        tk.Radiobutton(tab_frame, text="Original", variable=self.view_mode,
                      value='original', command=self.switch_view,
                      bg=COLORS['surface'], fg=COLORS['text_secondary'],
                      selectcolor=COLORS['primary'], font=FONTS['body'],
                      activebackground=COLORS['surface']).pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Radiobutton(tab_frame, text="Redacted", variable=self.view_mode,
                      value='redacted', command=self.switch_view,
                      bg=COLORS['surface'], fg=COLORS['text_secondary'],
                      selectcolor=COLORS['primary'], font=FONTS['body'],
                      activebackground=COLORS['surface']).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Image display card
        display_card = ModernCard(parent, padding=0)
        display_card.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display with scrollbars
        canvas_frame = tk.Frame(display_card.inner_frame, bg=COLORS['bg_secondary'])
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas
        self.image_canvas = tk.Canvas(canvas_frame, bg=COLORS['bg_secondary'],
                                     highlightthickness=0,
                                     yscrollcommand=v_scrollbar.set,
                                     xscrollcommand=h_scrollbar.set)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=self.image_canvas.yview)
        h_scrollbar.config(command=self.image_canvas.xview)
        
        # Placeholder text
        self.placeholder_text = self.image_canvas.create_text(
            500, 300, text="No document loaded\nUpload an image or PDF to begin",
            font=FONTS['subheader'], fill=COLORS['text_muted'], justify=tk.CENTER)
    
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        status_bar = tk.Frame(self, bg=COLORS['bg_secondary'], height=40)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(status_bar, text="Ready", 
                                     font=FONTS['caption'],
                                     bg=COLORS['bg_secondary'], 
                                     fg=COLORS['text_muted'])
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Version info
        tk.Label(status_bar, text="v3.0 Enterprise | © 2024 Privara Intellectus",
                font=FONTS['caption'], bg=COLORS['bg_secondary'],
                fg=COLORS['text_muted']).pack(side=tk.RIGHT, padx=20)
    
    # ===== EVENT HANDLERS =====
    
    def select_file(self):
        """Open file dialog to select image or PDF."""
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.pdf"),
                ("Images", "*.jpg *.jpeg *.png"),
                ("PDF", "*.pdf")
            ]
        )
        
        if file_path:
            if file_path.lower().endswith('.pdf'):
                self.load_pdf_from_path(file_path)
            else:
                self.load_image_from_path(file_path)
    
    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.load_image_from_path(file_path)
    
    def load_image_from_path(self, file_path):
        """Load image from given path."""
        try:
            self.update_status("Loading image...")
            
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image")
            
            self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.redacted_image = None
            self.current_file_path = file_path
            self.is_pdf_mode = False
            
            # Update UI
            self.file_label.config(text=f"File: {Path(file_path).name}\nType: Image")
            self.display_image(self.current_image)
            self.redact_btn.set_state('normal')
            self.save_btn.set_state('disabled')
            self.export_audit_btn.set_state('disabled')
            
            self.update_status(f"Loaded: {Path(file_path).name}")
            logger.info(f"Image loaded: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            logger.error(f"Image load error: {e}")
            self.update_status("Error loading image")
    
    def load_pdf(self):
        """Load a PDF file."""
        file_path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF", "*.pdf")]
        )
        
        if file_path:
            self.load_pdf_from_path(file_path)
    
    def load_pdf_from_path(self, file_path):
        """Load PDF from given path."""
        try:
            self.update_status("Loading PDF...")
            
            # Basic validation
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.current_file_path = file_path
            self.is_pdf_mode = True
            self.current_image = None
            self.redacted_image = None
            
            # Update UI
            file_name = Path(file_path).name
            self.file_label.config(text=f"File: {file_name}\nType: PDF (Multi-page)")
            self.redact_btn.set_state('normal')
            self.save_btn.set_state('disabled')
            self.export_audit_btn.set_state('disabled')
            
            # Show PDF icon placeholder
            self.image_canvas.delete('all')
            self.image_canvas.create_text(
                500, 300, text=f"📄\n{file_name}\n\nPDF Ready for Processing\nClick 'Redact PII' to begin",
                font=FONTS['subheader'], fill=COLORS['text_secondary'], justify=tk.CENTER)
            
            self.update_status(f"PDF loaded: {file_name}")
            logger.info(f"PDF loaded: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
            logger.error(f"PDF load error: {e}")
            self.update_status("Error loading PDF")
    
    def redact_pii(self):
        """Process and redact PII from the loaded document."""
        if self.processing:
            return
        
        try:
            self.processing = True
            self.update_status("Processing document...")
            self.redact_btn.set_state('disabled')
            
            # Show processing animation
            progress = ProgressIndicator(self.image_canvas, size=100)
            progress_window = self.image_canvas.create_window(
                500, 300, window=progress)
            
            # Animate progress
            def animate_progress(current=0):
                if current <= 90:
                    progress.set_progress(current)
                    self.after(50, lambda: animate_progress(current + 5))
            
            animate_progress()
            self.update()
            
            if self.is_pdf_mode:
                # Process PDF
                result = self.pdf_redactor.redact_pdf(self.current_file_path)
                
                # Update stats
                self.stats['total_detections'] = result['summary']['total_pii_found']
                self.stats['high_risk'] = result['summary']['risk_breakdown']['HIGH']
                self.stats['medium_risk'] = result['summary']['risk_breakdown']['MEDIUM']
                self.stats['low_risk'] = result['summary']['risk_breakdown']['LOW']
                self.stats['processing_time'] = result['summary']['total_processing_time']
                
                self.audit_data = result
                self.current_file_path = result['output_pdf_path']
                
                # Show success message
                self.image_canvas.delete(progress_window)
                self.image_canvas.delete('all')
                self.image_canvas.create_text(
                    500, 300,
                    text=f"✅ PDF Successfully Redacted!\n\n{result['summary']['total_pages']} pages processed\n{result['summary']['total_pii_found']} PII items redacted\n\nTime: {result['summary']['total_processing_time']:.2f}s",
                    font=FONTS['subheader'], fill=COLORS['success'], justify=tk.CENTER)
                
            else:
                # Process image
                result = self.redactor.redact_image(self.current_image)
                
                # Update stats
                total_pii = len(result['detections'])
                high_risk = sum(1 for d in result['detections'] if d['risk_level'] == 'HIGH')
                medium_risk = sum(1 for d in result['detections'] if d['risk_level'] == 'MEDIUM')
                low_risk = sum(1 for d in result['detections'] if d['risk_level'] == 'LOW')
                
                self.stats['total_detections'] = total_pii
                self.stats['high_risk'] = high_risk
                self.stats['medium_risk'] = medium_risk
                self.stats['low_risk'] = low_risk
                self.stats['processing_time'] = result.get('processing_time', 0)
                
                self.redacted_image = result['redacted_image']
                self.audit_data = result
                
                # Display redacted image
                progress.set_progress(100)
                self.after(300, lambda: self.image_canvas.delete(progress_window))
                self.after(400, lambda: self.display_image(self.redacted_image))
                self.view_mode.set('redacted')
            
            # Update stats cards
            self.update_stats_display()
            
            # Enable buttons
            self.save_btn.set_state('normal')
            self.export_audit_btn.set_state('normal')
            self.redact_btn.set_state('normal')
            
            self.update_status(f"Processing complete - {self.stats['total_detections']} PII items redacted")
            logger.info(f"Redaction complete: {self.stats['total_detections']} detections")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            logger.error(f"Redaction error: {e}")
            self.update_status("Processing failed")
        finally:
            self.processing = False
    
    def save_result(self):
        """Save the redacted document."""
        if self.is_pdf_mode and self.audit_data:
            # PDF already saved, just show location
            output_path = self.audit_data.get('output_pdf_path')
            if output_path:
                messagebox.showinfo("Saved", f"Redacted PDF saved to:\n{output_path}")
                self.update_status(f"Saved: {Path(output_path).name}")
        elif self.redacted_image is not None:
            # Save image
            file_path = filedialog.asksaveasfilename(
                title="Save Redacted Image",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
            )
            
            if file_path:
                try:
                    # Convert RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(self.redacted_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file_path, image_bgr)
                    messagebox.showinfo("Saved", f"Redacted image saved to:\n{file_path}")
                    self.update_status(f"Saved: {Path(file_path).name}")
                    logger.info(f"Saved redacted image: {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save: {str(e)}")
                    logger.error(f"Save error: {e}")
        else:
            messagebox.showwarning("Nothing to Save", "Please process a document first.")
    
    def export_audit(self):
        """Export audit log to JSON."""
        if not self.audit_data:
            messagebox.showwarning("No Data", "No audit data available to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Audit Log",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.audit_data, f, indent=2, default=str)
                messagebox.showinfo("Exported", f"Audit log exported to:\n{file_path}")
                self.update_status(f"Exported audit: {Path(file_path).name}")
                logger.info(f"Exported audit log: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
                logger.error(f"Export error: {e}")
    
    def clear_all(self):
        """Clear all loaded data and reset UI."""
        self.current_image = None
        self.redacted_image = None
        self.audit_data = None
        self.current_file_path = None
        self.is_pdf_mode = False
        
        # Reset stats
        self.stats = {
            'total_detections': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'processing_time': 0
        }
        self.update_stats_display()
        
        # Reset UI
        self.file_label.config(text="No file loaded")
        self.redact_btn.set_state('disabled')
        self.save_btn.set_state('disabled')
        self.export_audit_btn.set_state('disabled')
        
        # Clear canvas
        self.image_canvas.delete('all')
        self.image_canvas.create_text(
            500, 300, text="No document loaded\nUpload an image or PDF to begin",
            font=FONTS['subheader'], fill=COLORS['text_muted'], justify=tk.CENTER)
        
        self.update_status("Ready")
        logger.info("UI cleared")
    
    def display_image(self, image):
        """Display image on canvas."""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Calculate scaling to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 1000
                canvas_height = 700
            
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(pil_image)
            
            # Display on canvas
            self.image_canvas.delete('all')
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.photo_image, anchor=tk.CENTER)
            
            # Update scroll region
            self.image_canvas.config(scrollregion=self.image_canvas.bbox('all'))
            
        except Exception as e:
            logger.error(f"Display error: {e}")
    
    def switch_view(self):
        """Switch between original and redacted view."""
        mode = self.view_mode.get()
        if mode == 'original' and self.current_image is not None:
            self.display_image(self.current_image)
        elif mode == 'redacted' and self.redacted_image is not None:
            self.display_image(self.redacted_image)
    
    def update_stats_display(self):
        """Update statistics cards with current values."""
        self.total_stat.update_value(self.stats['total_detections'])
        self.time_stat.update_value(f"{self.stats['processing_time']:.2f}")
        self.high_risk_stat.update_value(self.stats['high_risk'])
        self.medium_risk_stat.update_value(self.stats['medium_risk'])
    
    def update_status(self, message):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.update_idletasks()
    
    def view_audit_logs(self):
        """Open audit logs directory."""
        audit_dir = Path("output/audit_logs")
        if audit_dir.exists():
            import os
            os.startfile(audit_dir)
        else:
            messagebox.showinfo("No Logs", "No audit logs found.")
    
    def open_settings(self):
        """Open settings dialog (placeholder)."""
        messagebox.showinfo("Settings", "Settings panel coming soon!")


def main():
    """Launch the application."""
    app = PrivaraModernUI()
    app.mainloop()


if __name__ == "__main__":
    main()
