"""
PII Redactor MVP - Graphical User Interface
Company: Blackbox & Co
Features: Threaded processing, audit logging, explanations, watermarked redaction
"""

import os
import sys
import json
import threading
from pathlib import Path
from queue import Queue

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.ocr import TrOCRExtractor
    from src.detector import VisualPIIDetector
    from src.redactor import PIIRedactor
    REDACTOR_AVAILABLE = True
except ImportError:
    try:
        from ocr import TrOCRExtractor
        from detector import VisualPIIDetector
        from redactor import PIIRedactor
        REDACTOR_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import PIIRedactor: {e}")
        PIIRedactor = None
        REDACTOR_AVAILABLE = False


class PIIRedactorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PII Redactor MVP - Shyam & Co")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Initialize variables first
        self.current_image_path = None
        self.original_image = None  # Store original full-resolution image
        self.redacted_image = None
        self.audit_log = []  # List to store audit trail messages
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        self.pii_redactor = None  # Initialize as None first
        self.pii_redactor_ready = False
        
        # Threading for background operations
        self.result_queue = Queue()
        self.processing_thread = None
        self.current_audit_data = None  # Store latest audit data for explanations
        
        # Create blank image for initial display
        self.blank_image = Image.new('RGB', (400, 400), (240, 240, 240))
        self.blank_photo = ImageTk.PhotoImage(self.blank_image)
        
        # Load settings
        self.load_settings()
        
        # Create UI elements FIRST (this creates status_var)
        self.create_widgets()
        
        # Update status bar initially
        self.update_status("Initializing PII Redactor...")
        
        # Initialize PIIRedactor after GUI is created
        self.root.after(100, self.start_pii_redactor_initialization)
        
        # Start checking for results
        self.check_background_results()
    
    def start_pii_redactor_initialization(self):
        """Start PIIRedactor initialization in background thread."""
        if not REDACTOR_AVAILABLE or PIIRedactor is None:
            self.update_status("PII Redactor modules not available - Basic image viewing only")
            messagebox.showwarning(
                "Limited Functionality", 
                "PII detection modules are not available.\n\n"
                "You can still upload and view images, but PII redaction will not work.\n\n"
                "Please ensure all dependencies are installed:\n"
                "- transformers\n- ultralytics\n- torch\n- pytesseract"
            )
            if hasattr(self, 'redact_btn'):
                self.redact_btn.config(state=tk.DISABLED)
            return
        
        self.update_status("Loading AI models in background... This may take a few moments.")
        
        # Start initialization in background thread
        init_thread = threading.Thread(target=self.initialize_pii_redactor_background, daemon=True)
        init_thread.start()
    
    def initialize_pii_redactor_background(self):
        """Initialize PIIRedactor in background thread."""
        try:
            # This runs in background thread
            pii_redactor = PIIRedactor()
            self.result_queue.put(("init_success", pii_redactor))
        except Exception as e:
            self.result_queue.put(("init_error", str(e)))
    
    def check_background_results(self):
        """Check for results from background operations."""
        try:
            while not self.result_queue.empty():
                result_type, result_data = self.result_queue.get_nowait()
                
                if result_type == "init_success":
                    self.pii_redactor = result_data
                    self.pii_redactor_ready = True
                    self.update_status("✓ PII Redactor ready - Upload a document to begin")
                    
                elif result_type == "init_error":
                    error_msg = result_data
                    self.update_status("⚠ Failed to initialize PII Redactor - Image viewing available")
                    messagebox.showwarning(
                        "Initialization Warning", 
                        f"PII Redactor could not be fully initialized:\n\n{error_msg}\n\n"
                        "You can still upload and view images, but PII detection will not be available.\n\n"
                        "Common issues:\n"
                        "- Missing model files\n- Insufficient memory\n- Missing dependencies\n\n"
                        "Try restarting the application or check the console for details."
                    )
                    self.pii_redactor = None
                    if hasattr(self, 'redact_btn'):
                        self.redact_btn.config(state=tk.DISABLED)
                
                elif result_type == "redaction_success":
                    redacted_image, audit_data = result_data
                    self.redacted_image = redacted_image
                    self.current_audit_data = audit_data  # Store for explanations
                    self.process_audit_data(audit_data)
                    self.display_redacted_image()
                    self.save_btn.config(state=tk.NORMAL)
                    self.explain_btn.config(state=tk.NORMAL)  # Enable explanation button
                    self.update_status("✓ PII redaction completed successfully")
                    self.redact_btn.config(state=tk.NORMAL)
                    self.progress.stop()
                    self.progress.grid_remove()
                    
                elif result_type == "redaction_error":
                    error_msg = result_data
                    messagebox.showerror("Error", f"Failed to redact PII:\n\n{error_msg}")
                    self.update_status("✗ Error in PII redaction")
                    self.redact_btn.config(state=tk.NORMAL)
                    self.progress.stop()
                    self.progress.grid_remove()
                    
        except:
            pass  # Queue is empty
        
        # Schedule next check
        self.root.after(100, self.check_background_results)
    
    def process_audit_data(self, audit_data):
        """Process audit data from background redaction."""
        if audit_data:
            text_pii = audit_data.get('text_pii_count', 0)
            visual_pii = audit_data.get('visual_pii_count', 0)
            total_pii = audit_data.get('total_pii_detected', text_pii + visual_pii)
            pii_types = audit_data.get('detected_pii_types', [])
            
            audit_message = f"✓ Document processed: {total_pii} PII detections found"
            if text_pii > 0:
                audit_message += f" (Text: {text_pii}"
            if visual_pii > 0:
                audit_message += f", Visual: {visual_pii}"
            if text_pii > 0 or visual_pii > 0:
                audit_message += ")"
            if pii_types:
                audit_message += f"\n  Types detected: {', '.join(pii_types)}"
            
            self.audit_log.append(audit_message)
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)  # Status bar row
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Make preview area expandable
        
        # Title label
        title_label = ttk.Label(main_frame, text="PII Redactor MVP", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        company_label = ttk.Label(main_frame, text="Powered by Shyam & Co", font=("Arial", 10, "italic"), foreground="gray")
        company_label.grid(row=0, column=0, columnspan=2, pady=(35, 0))
        
        # Button frame for better layout
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        # Upload button
        upload_btn = ttk.Button(button_frame, text="📁 Upload Document", command=self.upload_document, padding=10)
        upload_btn.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        self.create_tooltip(upload_btn, "Upload a document to redact PII")
        
        # Redact button
        self.redact_btn = ttk.Button(button_frame, text="🔒 Redact PII", command=self.redact_pii, state=tk.DISABLED, padding=10)
        self.redact_btn.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.create_tooltip(self.redact_btn, "Redact PII from the uploaded document")
        
        # Image preview area (500x400)
        preview_frame = ttk.LabelFrame(main_frame, text="Document Preview", padding=10)
        preview_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.preview_label = ttk.Label(
            preview_frame, 
            image=self.blank_photo,
            background="lightgray"
        )
        self.preview_label.pack(expand=True)
        self.preview_label.image = self.blank_photo  # Keep reference
        
        # Action buttons frame
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=1)
        
        # Save button
        self.save_btn = ttk.Button(action_frame, text="💾 Save Redacted", command=self.save_redacted, state=tk.DISABLED, padding=8)
        self.save_btn.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        self.create_tooltip(self.save_btn, "Save the redacted document")
        
        # View Explanation button
        self.explain_btn = ttk.Button(action_frame, text="📋 View Explanations", command=self.view_explanations, state=tk.DISABLED, padding=8)
        self.explain_btn.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.create_tooltip(self.explain_btn, "View detailed explanations for detected PII")
        
        # View Log button  
        log_btn = ttk.Button(action_frame, text="📜 View Audit Log", command=self.view_log, padding=8)
        log_btn.grid(row=0, column=2, padx=5, sticky=(tk.W, tk.E))
        self.create_tooltip(log_btn, "View the audit trail")
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient='horizontal', length=300, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        self.progress.grid_remove()  # Hide progress bar initially
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Document", command=self.upload_document, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Redacted", command=self.save_redacted, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Clear Recent Files", command=self.clear_recent_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Alt+F4")
        
        # Recent files submenu
        self.recent_files = []
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        self.update_recent_menu()
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        tooltip = ttk.Label(self.root, text=text, background="#ffffe0", 
                          relief="solid", borderwidth=1, padding=5)
        tooltip.place_forget()
        
        def enter(event):
            x = widget.winfo_rootx() + widget.winfo_width() // 2
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip.place(x=x, y=y, anchor=tk.N)
            
        def leave(event):
            tooltip.place_forget()
            
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def upload_document(self):
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            if self.pii_redactor_ready:
                self.redact_btn.config(state=tk.NORMAL)
            self.update_status(f"✓ Document loaded: {os.path.basename(file_path)}")
            self.audit_log.append(f"📁 Document uploaded: {file_path}")
            
            # Add to recent files
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
            self.recent_files.insert(0, file_path)
            self.recent_files = self.recent_files[:5]  # Keep only last 5
            self.update_recent_menu()
            self.save_settings()
    
    def display_image(self, image_path):
        try:
            self.original_image_path = image_path  # Store original full-resolution image path separately
            self.original_image = Image.open(image_path)  # Store original full-resolution image
            img = self.original_image.copy()  # Keep preview thumbnail for UI display ONLY
            img.thumbnail((500, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.preview_label.config(image=img_tk)
            self.preview_label.image = img_tk  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n\n{str(e)}")
            self.update_status("✗ Error loading image")

    
    def redact_pii(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload a document first.")
            return
        
        if not self.pii_redactor_ready or not self.pii_redactor:
            if not self.pii_redactor_ready:
                messagebox.showinfo(
                    "Please Wait", 
                    "PII Redactor is still initializing.\n\n"
                    "Please wait for the initialization to complete and try again."
                )
            else:
                messagebox.showerror(
                    "PII Redactor Not Available", 
                    "PII Redactor could not be initialized.\n\n"
                    "Please check the console output for details."
                )
            return
        
        # Check if already processing
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Processing", "PII redaction is already in progress. Please wait.")
            return
        
        self.update_status("⏳ Processing PII redaction in background...")
        self.redact_btn.config(state=tk.DISABLED)
        self.progress.grid()  # Show progress bar
        self.progress.start()
        
        # Start redaction in background thread
        self.processing_thread = threading.Thread(
            target=self.redact_pii_background, 
            args=(self.current_image_path,), 
            daemon=True
        )
        self.processing_thread.start()
    
    def redact_pii_background(self, image_path):
        """Perform PII redaction in background thread."""
        try:
            # ALWAYS use self.current_image_path (original file path)
            # Never use any PIL Image object from preview
            redacted_image = self.pii_redactor.process_document(self.current_image_path)
            
            # Get audit data from the redactor
            audit_data = self.pii_redactor.generate_audit_log()
            
            # Send results back to main thread
            self.result_queue.put(("redaction_success", (redacted_image, audit_data)))
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Redaction error:\n{error_details}")
            self.result_queue.put(("redaction_error", str(e)))
    
    def display_redacted_image(self):
        if self.redacted_image:
            # Load the FULL redacted image from PIIRedactor
            img = self.redacted_image.copy()
            # THEN thumbnail it for display only
            img.thumbnail((500, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.preview_label.config(image=img_tk)
            self.preview_label.image = img_tk
            # Ensure saved image is full resolution - it is, since self.redacted_image is full
    
    def save_redacted(self):
        if not self.redacted_image:
            messagebox.showwarning("Warning", "No redacted image to save.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Redacted Document",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if save_path:
            try:
                self.redacted_image.save(save_path)
                self.update_status(f"✓ Redacted document saved: {os.path.basename(save_path)}")
                self.audit_log.append(f"💾 Redacted document saved: {save_path}")
                messagebox.showinfo("Success", "Redacted document saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n\n{str(e)}")
                self.update_status("✗ Error saving document")
    
    def view_log(self):
        log_window = tk.Toplevel(self.root)
        log_window.title("Audit Trail - Shyam & Co")
        log_window.geometry("600x450")
        self.center_window(log_window, 600, 450)
        
        # Create frame for text and scrollbar
        frame = ttk.Frame(log_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        log_text = tk.Text(frame, wrap=tk.WORD, padx=10, pady=10, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=log_text.yview)
        log_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.pack(fill=tk.BOTH, expand=True)
        
        log_text.insert(tk.END, "📜 AUDIT TRAIL\n")
        log_text.insert(tk.END, "=" * 60 + "\n\n")
        
        for entry in self.audit_log:
            log_text.insert(tk.END, entry + "\n\n")
        
        if not self.audit_log:
            log_text.insert(tk.END, "No audit entries yet. Upload and redact a document to begin.\n")
        
        log_text.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ttk.Button(frame, text="Close", command=log_window.destroy, padding=5)
        close_btn.pack(pady=(10, 0))
    
    def view_explanations(self):
        """Show detailed explanations for detected PII."""
        if not self.current_audit_data:
            messagebox.showinfo("No Explanations", "No PII explanations available. Please redact a document first.")
            return
        
        explain_window = tk.Toplevel(self.root)
        explain_window.title("PII Detection Explanations - Shyam & Co")
        self.center_window(explain_window, 750, 550)
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(explain_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable text widget
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        explain_text = tk.Text(text_frame, wrap=tk.WORD, padx=15, pady=15, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=explain_text.yview)
        explain_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        explain_text.pack(fill=tk.BOTH, expand=True)
        
        # Build explanation content
        content = []
        content.append("🔍 PII DETECTION EXPLANATIONS\n")
        content.append("=" * 70 + "\n\n")
        
        content.append(f"📄 Document: {self.current_audit_data.get('original_filename', 'Unknown')}\n")
        content.append(f"⏰ Processed: {self.current_audit_data.get('timestamp', 'Unknown')}\n")
        content.append(f"🏢 Company Watermark: {self.current_audit_data.get('company_watermark', 'Privara & Co')}\n")
        content.append(f"🔒 Redaction Method: {self.current_audit_data.get('redaction_method', 'Watermarked Black Box')}\n\n")
        
        text_pii = self.current_audit_data.get('text_pii_count', 0)
        visual_pii = self.current_audit_data.get('visual_pii_count', 0)
        total = self.current_audit_data.get('total_pii_detected', text_pii + visual_pii)
        
        content.append(f"📊 Summary Statistics:\n")
        content.append(f"   • Total PII Detected: {total}\n")
        content.append(f"   • Text-based PII: {text_pii}\n")
        content.append(f"   • Visual PII: {visual_pii}\n\n")
        
        pii_types = self.current_audit_data.get('detected_pii_types', [])
        if pii_types:
            content.append(f"🏷️  Detected PII Types:\n")
            for pii_type in pii_types:
                content.append(f"   • {pii_type}\n")
            content.append("\n")
        
        content.append("─" * 70 + "\n\n")
        content.append("💡 Privacy Protection Features:\n\n")
        content.append(f"   • Differential Privacy (ε): {self.current_audit_data.get('epsilon_used', 'N/A')}\n")
        content.append(f"   • Data Region: {self.current_audit_data.get('data_region', 'N/A')}\n")
        content.append(f"   • Watermark Applied: Yes (Privara & Co)\n")
        content.append(f"   • Full Opacity Redaction: Yes\n\n")
        
        # Insert content
        explain_text.insert(tk.END, "".join(content))
        explain_text.config(state=tk.DISABLED)
        
        # Add close button
        close_btn = ttk.Button(main_frame, text="Close", command=explain_window.destroy, padding=8)
        close_btn.pack(pady=(10, 0))
    
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About PII Redactor")
        self.center_window(about_window, 400, 250)
        
        about_frame = ttk.Frame(about_window, padding=20)
        about_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(about_frame, text="PII Redactor MVP", font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(about_frame, text="Version 1.0.0", font=("Arial", 10)).pack()
        ttk.Label(about_frame, text="Powered by Shyam & Co", font=("Arial", 10, "italic")).pack(pady=5)
        ttk.Label(about_frame, text="\nPrivacy-First Document Redaction\nwith AI-Powered Detection", 
                 justify=tk.CENTER).pack(pady=10)
        ttk.Label(about_frame, text="© 2025 Shyam & Co. All rights reserved.", 
                 font=("Arial", 8), foreground="gray").pack(pady=10)
        
        ttk.Button(about_frame, text="Close", command=about_window.destroy, padding=5).pack(pady=10)
    
    def show_documentation(self):
        messagebox.showinfo(
            "Documentation",
            "PII Redactor MVP - Quick Guide\n\n"
            "1. Upload Document: Click 'Upload Document' to select an image\n"
            "2. Redact PII: Click 'Redact PII' to process the document\n"
            "3. Save: Click 'Save Redacted' to export the result\n"
            "4. View Explanations: See detailed detection information\n"
            "5. Audit Log: Track all operations performed\n\n"
            "Supported Formats: PNG, JPG, JPEG, BMP, TIFF\n\n"
            "For support, contact: sundxrr@gmail.com"
        )
    
    def update_status(self, message):
        """Update status bar with proper error handling."""
        try:
            if hasattr(self, 'status_var') and self.status_var:
                self.status_var.set(message)
        except Exception:
            pass
    
    def center_window(self, window, width, height):
        """Center a window on the screen"""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        
        window.geometry(f"{width}x{height}+{x}+{y}")
    
    def update_recent_menu(self):
        self.recent_menu.delete(0, tk.END)
        for file_path in self.recent_files:
            display_name = os.path.basename(file_path)
            self.recent_menu.add_command(
                label=display_name, 
                command=lambda path=file_path: self.open_recent_file(path)
            )
        if not self.recent_files:
            self.recent_menu.add_command(label="(No recent files)", state=tk.DISABLED)
    
    def open_recent_file(self, file_path):
        """Open a file from the recent files list"""
        if os.path.exists(file_path):
            self.current_image_path = file_path
            self.display_image(file_path)
            if self.pii_redactor_ready:
                self.redact_btn.config(state=tk.NORMAL)
            self.update_status(f"✓ Document loaded: {os.path.basename(file_path)}")
            self.audit_log.append(f"📁 Document opened from recent: {file_path}")
            
            # Move to top of recent list
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
            self.recent_files.insert(0, file_path)
            self.update_recent_menu()
            self.save_settings()
        else:
            messagebox.showerror("File Not Found", f"The file was not found:\n\n{file_path}")
            self.recent_files.remove(file_path)
            self.update_recent_menu()
            self.save_settings()
    
    def clear_recent_files(self):
        """Clear the recent files list"""
        self.recent_files = []
        self.update_recent_menu()
        self.save_settings()
        messagebox.showinfo("Cleared", "Recent files list cleared.")
    
    def load_settings(self):
        """Load application settings"""
        self.recent_files = []  # Initialize first
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self.recent_files = settings.get("recent_files", [])
        except Exception as e:
            print(f"Could not load settings: {e}")
            self.recent_files = []
    
    def save_settings(self):
        """Save application settings"""
        try:
            if not hasattr(self, 'recent_files'):
                self.recent_files = []
            
            settings = {
                "recent_files": self.recent_files
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Could not save settings: {e}")
    
    def __del__(self):
        try:
            self.save_settings()
        except:
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    root = tk.Tk()
    app = PIIRedactorGUI(root)
    root.mainloop()
