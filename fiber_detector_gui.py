import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import json
import threading
from fiber_detector import FiberLengthDetector
import os
import cv2
import numpy as np

class EnhancedFiberDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fiber Length Detector - Enhanced Vision Interface")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f0f2f5')
        
        # Initialize detector and UI state
        self.detector = None
        self.selected_files = []
        self.mode = "single"
        self.current_result = None
        self.image_panels = {}  # Store image display panels
        
        self.create_enhanced_widgets()
        self.initialize_detector()
    
    def create_enhanced_widgets(self):
        # Title Section with Enhanced Styling
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="Fiber Length Detector - Enhanced Vision",
                              font=("Segoe UI", 24, "bold"),
                              fg="#ecf0f1", bg="#2c3e50")
        title_label.pack(expand=True, pady=10)
        
        subtitle_label = tk.Label(title_frame,
                                 text="AI-Powered Measurement Extraction with Visual Feedback",
                                 font=("Segoe UI", 12),
                                 fg="#bdc3c7", bg="#2c3e50")
        subtitle_label.pack()
        
        # Main Container with Scrollable Canvas
        self.main_canvas = tk.Canvas(self.root, bg='#f0f2f5')
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg='#f0f2f5')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.main_canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        self.scrollbar.pack(side="right", fill="y")
        
        # Status Section
        status_frame = tk.Frame(self.scrollable_frame, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.status_label = tk.Label(status_frame,
                                   text="Initializing AI vision model...",
                                   font=("Segoe UI", 11, "bold"),
                                   fg="#3498db", bg='#ecf0f1')
        self.status_label.pack(pady=15)
        
        # Control Panel
        control_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="Control Panel",
                                    font=("Segoe UI", 14, "bold"),
                                    bg='#f0f2f5', fg='#2c3e50')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Mode Selection with Enhanced Styling
        mode_frame = tk.Frame(control_frame, bg='#f0f2f5')
        mode_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(mode_frame, text="Analysis Mode:", 
                font=("Segoe UI", 12, "bold"), bg='#f0f2f5').pack(anchor=tk.W)
        
        self.mode_var = tk.StringVar(value="single")
        
        mode_buttons_frame = tk.Frame(mode_frame, bg='#f0f2f5')
        mode_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        single_radio = tk.Radiobutton(mode_buttons_frame,
                                     text="Single Image Analysis",
                                     variable=self.mode_var,
                                     value="single",
                                     command=self.on_mode_change,
                                     font=("Segoe UI", 10),
                                     bg='#f0f2f5', fg='#2c3e50')
        single_radio.pack(side=tk.LEFT, padx=(0, 30))
        
        dual_radio = tk.Radiobutton(mode_buttons_frame,
                                   text="Dual Image Comparison",
                                   variable=self.mode_var,
                                   value="dual",
                                   command=self.on_mode_change,
                                   font=("Segoe UI", 10),
                                   bg='#f0f2f5', fg='#2c3e50')
        dual_radio.pack(side=tk.LEFT)
        
        # File Selection with Enhanced Interface
        file_frame = tk.Frame(control_frame, bg='#f0f2f5')
        file_frame.pack(fill=tk.X, padx=20, pady=15)
        
        select_btn = tk.Button(file_frame,
                              text="Select Image File(s)",
                              command=self.select_files,
                              font=("Segoe UI", 12, "bold"),
                              bg="#3498db", fg="white",
                              padx=30, pady=12,
                              cursor="hand2",
                              relief=tk.FLAT)
        select_btn.pack(pady=10)
        
        # Process Button with Enhanced Styling
        self.process_btn = tk.Button(file_frame,
                                   text="Analyze Image(s)",
                                   command=self.process_images_threaded,
                                   font=("Segoe UI", 14, "bold"),
                                   bg="#27ae60", fg="white",
                                   padx=40, pady=15,
                                   cursor="hand2",
                                   state=tk.DISABLED,
                                   relief=tk.FLAT)
        self.process_btn.pack(pady=10)
        
        # Progress Bar with Enhanced Styling
        self.progress = ttk.Progressbar(file_frame,
                                       mode='indeterminate',
                                       length=500,
                                       style='Enhanced.Horizontal.TProgressbar')
        self.progress.pack(pady=15)
        
        # Image Display Area
        self.image_display_frame = tk.Frame(self.scrollable_frame, bg='#f0f2f5')
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Results Section with Enhanced Styling
        results_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="Analysis Results",
                                    font=("Segoe UI", 14, "bold"),
                                    bg='#f0f2f5', fg='#2c3e50')
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.results_text = scrolledtext.ScrolledText(results_frame,
                                                     font=("Consolas", 11),
                                                     wrap=tk.WORD,
                                                     height=15,
                                                     bg='#ffffff',
                                                     fg='#2c3e50')
        self.results_text.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        # Action Buttons
        button_frame = tk.Frame(self.scrollable_frame, bg='#f0f2f5')
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        clear_btn = tk.Button(button_frame,
                             text="Clear Results",
                             command=self.clear_results,
                             font=("Segoe UI", 10, "bold"),
                             bg="#e74c3c", fg="white",
                             padx=20, pady=8,
                             relief=tk.FLAT)
        clear_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        save_btn = tk.Button(button_frame,
                            text="Save Results",
                            command=self.save_results,
                            font=("Segoe UI", 10, "bold"),
                            bg="#f39c12", fg="white",
                            padx=20, pady=8,
                            relief=tk.FLAT)
        save_btn.pack(side=tk.LEFT)
        
        # Configure custom styles
        self.configure_styles()
        
        # Bind mouse wheel to canvas
        self.main_canvas.bind("<MouseWheel>", self.on_mousewheel)
    
    def configure_styles(self):
        """Configure custom TTK styles"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Enhanced.Horizontal.TProgressbar',
                       background='#3498db',
                       troughcolor='#ecf0f1',
                       borderwidth=0,
                       lightcolor='#3498db',
                       darkcolor='#3498db')
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def initialize_detector(self):
        """Initialize the detector in a separate thread"""
        def init_thread():
            try:
                self.detector = FiberLengthDetector(model_name='llava-phi3')
                self.root.after(0, self.on_detector_ready)
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.on_detector_error(msg))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def on_detector_ready(self):
        """Called when detector is ready"""
        self.status_label.config(text="AI vision model ready for analysis", fg="#27ae60")
        self.update_process_button_state()
    
    def on_detector_error(self, error_msg):
        """Called when detector fails to initialize"""
        self.status_label.config(text="Initialization failed - Check console for details", fg="#e74c3c")
        messagebox.showerror("Initialization Error", 
                           f"Failed to initialize AI model:\n\n{error_msg}")
    
    def on_mode_change(self):
        """Called when mode is changed"""
        self.mode = self.mode_var.get()
        self.selected_files = []
        self.clear_image_displays()
        self.update_process_button_state()
    
    def select_files(self):
        """Select image files based on mode"""
        filetypes = [
            ("All Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        if self.mode == "dual":
            files = filedialog.askopenfilenames(
                title="Select 2 Fiber Images for Comparison",
                filetypes=filetypes
            )
            if len(files) != 2:
                messagebox.showwarning("File Selection", 
                                     "Please select exactly 2 images for comparison mode!")
                return
            self.selected_files = list(files)
        else:
            file = filedialog.askopenfilename(
                title="Select Fiber Image",
                filetypes=filetypes
            )
            if file:
                self.selected_files = [file]
        
        if self.selected_files:
            self.display_selected_images()
        self.update_process_button_state()
    
    def display_selected_images(self):
        """Display the selected images in the interface"""
        self.clear_image_displays()
        
        if not self.selected_files:
            return
        
        # Create title for image section
        title_label = tk.Label(self.image_display_frame,
                              text=f"Selected Images ({len(self.selected_files)})",
                              font=("Segoe UI", 16, "bold"),
                              bg='#f0f2f5', fg='#2c3e50')
        title_label.pack(pady=(10, 20))
        
        if self.mode == "dual":
            # Side by side for dual mode
            images_container = tk.Frame(self.image_display_frame, bg='#f0f2f5')
            images_container.pack(fill=tk.X, padx=20)
            
            for i, file_path in enumerate(self.selected_files):
                self.create_image_panel(images_container, file_path, f"Image {i+1}", 
                                      side=tk.LEFT if i == 0 else tk.RIGHT)
        else:
            # Single image display
            self.create_image_panel(self.image_display_frame, self.selected_files[0], 
                                  "Selected Image")
    
    def create_image_panel(self, parent, file_path, title, side=None):
        """Create an image display panel with title and results area"""
        # Create container
        if side:
            container = tk.Frame(parent, bg='#ffffff', relief=tk.RAISED, bd=2)
            container.pack(side=side, fill=tk.BOTH, expand=True, padx=10)
        else:
            container = tk.Frame(parent, bg='#ffffff', relief=tk.RAISED, bd=2)
            container.pack(fill=tk.X, padx=20, pady=10)
        
        # Title
        title_frame = tk.Frame(container, bg='#34495e')
        title_frame.pack(fill=tk.X)
        
        tk.Label(title_frame, text=title, 
                font=("Segoe UI", 12, "bold"),
                fg="white", bg="#34495e").pack(pady=8)
        
        # File name
        filename = os.path.basename(file_path)
        tk.Label(title_frame, text=f"File: {filename}", 
                font=("Segoe UI", 9),
                fg="#bdc3c7", bg="#34495e").pack(pady=(0, 8))
        
        # Image display
        try:
            # Load and resize image
            pil_image = Image.open(file_path)
            display_size = (400, 300) if side else (600, 400)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            
            image_label = tk.Label(container, image=photo, bg='#ffffff')
            image_label.image = photo  # Keep a reference
            image_label.pack(pady=15)
            
            # Results area for this image
            results_frame = tk.Frame(container, bg='#ecf0f1')
            results_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
            
            tk.Label(results_frame, text="Analysis Results:", 
                    font=("Segoe UI", 11, "bold"),
                    bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, pady=(10, 5))
            
            result_text = tk.Text(results_frame, height=6, font=("Consolas", 9),
                                bg='#ffffff', fg='#2c3e50', wrap=tk.WORD)
            result_text.pack(fill=tk.X, padx=10, pady=(0, 10))
            result_text.insert('1.0', "Analysis pending...")
            result_text.config(state=tk.DISABLED)
            
            # Store reference for later updates
            self.image_panels[file_path] = {
                'container': container,
                'result_text': result_text,
                'title': title
            }
            
        except Exception as e:
            error_label = tk.Label(container, 
                                 text=f"Error loading image: {str(e)}", 
                                 fg="red", bg='#ffffff')
            error_label.pack(pady=20)
    
    def clear_image_displays(self):
        """Clear all image displays"""
        for widget in self.image_display_frame.winfo_children():
            widget.destroy()
        self.image_panels.clear()
    
    def update_process_button_state(self):
        """Update the process button state based on current conditions"""
        if not self.detector:
            self.process_btn.config(state=tk.DISABLED)
            return
        
        required_files = 2 if self.mode == "dual" else 1
        if len(self.selected_files) == required_files:
            self.process_btn.config(state=tk.NORMAL)
        else:
            self.process_btn.config(state=tk.DISABLED)
    
    def process_images_threaded(self):
        """Process images in a separate thread"""
        if not self.selected_files or not self.detector:
            messagebox.showwarning("Not Ready", "Please select images and ensure AI model is ready!")
            return
        
        # Start processing
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Processing images with AI vision...", fg="#3498db")
        
        # Clear previous individual results
        for file_path, panel_info in self.image_panels.items():
            result_text = panel_info['result_text']
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert('1.0', "Processing...")
            result_text.config(state=tk.DISABLED)
        
        def process_thread():
            try:
                if self.mode == "dual" and len(self.selected_files) == 2:
                    result = self.detector.process_two_images(
                        self.selected_files[0], 
                        self.selected_files[1]
                    )
                else:
                    result = self.detector.process_image(self.selected_files[0])
                
                self.root.after(0, lambda: self.on_process_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self.on_process_error(str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def on_process_complete(self, result):
        """Called when processing is complete"""
        self.progress.stop()
        self.update_process_button_state()
        self.current_result = result
        
        if result and 'error' not in result:
            self.status_label.config(text="Analysis complete!", fg="#27ae60")
            self.display_results(result)
            self.update_individual_results(result)
        else:
            self.status_label.config(text="Analysis failed", fg="#e74c3c")
            error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Failed to process image(s).\nError: {error_msg}\n")
    
    def on_process_error(self, error_msg):
        """Called when processing fails"""
        self.progress.stop()
        self.update_process_button_state()
        self.status_label.config(text="Processing error", fg="#e74c3c")
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error occurred during processing:\n{error_msg}")
        
        # Update individual result panels
        for file_path, panel_info in self.image_panels.items():
            result_text = panel_info['result_text']
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert('1.0', f"Processing failed:\n{error_msg}")
            result_text.config(state=tk.DISABLED)
    
    def update_individual_results(self, result):
        """Update individual image result panels"""
        if self.mode == "dual":
            # Update both image panels for dual mode
            img1_result = result.get('image1_result', {})
            img2_result = result.get('image2_result', {})
            
            results_map = {
                self.selected_files[0]: img1_result,
                self.selected_files[1]: img2_result
            }
            
            for file_path, img_result in results_map.items():
                if file_path in self.image_panels:
                    self.update_single_result_panel(file_path, img_result)
        else:
            # Update single image panel
            if self.selected_files[0] in self.image_panels:
                self.update_single_result_panel(self.selected_files[0], result)
    
    def update_single_result_panel(self, file_path, img_result):
        """Update a single result panel with analysis results"""
        if file_path not in self.image_panels:
            return
        
        result_text = self.image_panels[file_path]['result_text']
        result_text.config(state=tk.NORMAL)
        result_text.delete('1.0', tk.END)
        
        # Format results
        length = img_result.get('detected_length', 'Not detected')
        unit = img_result.get('unit', 'N/A')
        confidence = img_result.get('confidence', 0)
        
        if length and length != 'Not detected':
            result_text.insert(tk.END, f"Detected Length: {length} {unit}\n")
            result_text.insert(tk.END, f"Confidence: {confidence}%\n")
            
            if confidence >= 80:
                status = "HIGH CONFIDENCE"
            elif confidence >= 50:
                status = "MEDIUM CONFIDENCE"
            else:
                status = "LOW CONFIDENCE"
            
            result_text.insert(tk.END, f"Status: {status}\n\n")
        else:
            result_text.insert(tk.END, "NO LENGTH DETECTED\n\n")
        
        # Additional info
        method = img_result.get('method', 'N/A')
        result_text.insert(tk.END, f"Method: {method}\n")
        
        additional = img_result.get('additional_numbers', [])
        if additional:
            result_text.insert(tk.END, f"Other Numbers: {', '.join(map(str, additional))}\n")
        
        result_text.config(state=tk.DISABLED)
    
    def display_results(self, result):
        """Display the analysis results in main results area"""
        self.results_text.delete(1.0, tk.END)
        
        if self.mode == "dual":
            self.display_dual_results(result)
        else:
            self.display_single_results(result)
    
    def display_single_results(self, result):
        """Display single image analysis results"""
        self.results_text.insert(tk.END, "SINGLE IMAGE ANALYSIS RESULTS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        length = result.get('detected_length', 'Not detected')
        unit = result.get('unit', 'N/A')
        confidence = result.get('confidence', 0)
        
        if length and length != 'Not detected':
            self.results_text.insert(tk.END, f"DETECTED LENGTH: {length} {unit}\n")
            self.results_text.insert(tk.END, f"CONFIDENCE: {confidence}%\n")
            
            if confidence >= 80:
                status = "HIGH CONFIDENCE"
            elif confidence >= 50:
                status = "MEDIUM CONFIDENCE"
            else:
                status = "LOW CONFIDENCE"
            
            self.results_text.insert(tk.END, f"STATUS: {status}\n\n")
        else:
            self.results_text.insert(tk.END, "NO LENGTH DETECTED\n\n")
        
        # Additional details
        self.results_text.insert(tk.END, "ADDITIONAL DETAILS:\n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")
        self.results_text.insert(tk.END, f"Method: {result.get('method', 'N/A')}\n")
        self.results_text.insert(tk.END, f"Model Used: {result.get('model_used', 'N/A')}\n")
        
        raw_text = result.get('raw_text', '')
        if raw_text and raw_text != 'N/A':
            self.results_text.insert(tk.END, f"\nRaw AI Response:\n{raw_text}\n")
        
        additional = result.get('additional_numbers', [])
        if additional:
            self.results_text.insert(tk.END, f"\nOther Numbers Found: {', '.join(map(str, additional))}\n")
        
        # Full JSON
        self.results_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.results_text.insert(tk.END, "FULL JSON RESULT:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, json.dumps(result, indent=2))
    
    def display_dual_results(self, result):
        """Display dual image comparison results"""
        self.results_text.insert(tk.END, "DUAL IMAGE COMPARISON RESULTS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        difference = result.get('difference')
        if difference is not None:
            self.results_text.insert(tk.END, f"FIBER LENGTH DIFFERENCE: {difference} meters\n")
            
            diff_confidence = result.get('difference_confidence', 0)
            self.results_text.insert(tk.END, f"DIFFERENCE CONFIDENCE: {diff_confidence}%\n\n")
            
            if diff_confidence >= 80:
                status = "HIGH CONFIDENCE COMPARISON"
            elif diff_confidence >= 50:
                status = "MEDIUM CONFIDENCE COMPARISON"
            else:
                status = "LOW CONFIDENCE COMPARISON"
            
            self.results_text.insert(tk.END, f"STATUS: {status}\n\n")
        else:
            self.results_text.insert(tk.END, "COULD NOT CALCULATE DIFFERENCE\n\n")
        
        # Individual results summary
        img1_result = result.get('image1_result', {})
        img2_result = result.get('image2_result', {})
        
        self.results_text.insert(tk.END, "INDIVIDUAL RESULTS SUMMARY:\n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")
        
        # Image 1
        self.results_text.insert(tk.END, f"IMAGE 1:\n")
        length1 = img1_result.get('detected_length', 'Not detected')
        conf1 = img1_result.get('confidence', 0)
        self.results_text.insert(tk.END, f"   Length: {length1} meters\n")
        self.results_text.insert(tk.END, f"   Confidence: {conf1}%\n\n")
        
        # Image 2
        self.results_text.insert(tk.END, f"IMAGE 2:\n")
        length2 = img2_result.get('detected_length', 'Not detected')
        conf2 = img2_result.get('confidence', 0)
        self.results_text.insert(tk.END, f"   Length: {length2} meters\n")
        self.results_text.insert(tk.END, f"   Confidence: {conf2}%\n\n")
        
        # Full JSON
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, "FULL JSON RESULT:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, json.dumps(result, indent=2))
    
    def clear_results(self):
        """Clear the results display"""
        self.results_text.delete(1.0, tk.END)
        self.current_result = None
        
        # Clear individual result panels
        for file_path, panel_info in self.image_panels.items():
            result_text = panel_info['result_text']
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert('1.0', "Analysis pending...")
            result_text.config(state=tk.DISABLED)
        
        if self.detector:
            self.status_label.config(text="AI vision model ready for analysis", fg="#27ae60")
    
    def save_results(self):
        """Save results to JSON file"""
        if not self.current_result:
            messagebox.showwarning("No Results", "No results to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_result, f, indent=2)
                messagebox.showinfo("Saved", f"Results saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{e}")

def main():
    root = tk.Tk()
    app = EnhancedFiberDetectorGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
    