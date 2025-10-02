"""
GUI Interface for the Name Extractor AI Agent
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from name_extractor_agent import NameExtractorAgent

class NameExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Agent for Extracting Names from Images")
        self.root.geometry("600x400")
        
        # Initialize the agent
        self.agent = NameExtractorAgent()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Agent for Extracting Names from Images",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Image selection
        ttk.Label(main_frame, text="Select Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.image_path_var = tk.StringVar()
        image_entry = ttk.Entry(main_frame, textvariable=self.image_path_var, width=50)
        image_entry.grid(row=1, column=1, padx=(10, 5), pady=5)
        
        browse_button = ttk.Button(main_frame, text="Browse", command=self.browse_image)
        browse_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Output file selection
        ttk.Label(main_frame, text="Excel File Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_file_var = tk.StringVar(value="extracted_names.xlsx")
        output_entry = ttk.Entry(main_frame, textvariable=self.output_file_var, width=50)
        output_entry.grid(row=2, column=1, padx=(10, 5), pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Extract Names",
                                        command=self.process_image_threaded)
        self.process_button.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to work")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Results text area
        ttk.Label(main_frame, text="Results:").grid(row=6, column=0, sticky=tk.W, pady=(20, 5))
        
        # Create frame for text area and scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = tk.Text(text_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def browse_image(self):
        """Browse for image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.image_path_var.set(file_path)
    
    def process_image_threaded(self):
        """Process image in a separate thread to avoid freezing the GUI"""
        if not self.image_path_var.get():
            messagebox.showerror("Error", "Please select an image first")
            return

        if not Path(self.image_path_var.get()).exists():
            messagebox.showerror("Error", "File does not exist")
            return

        # Disable button and start progress
        self.process_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Processing...")
        self.results_text.delete(1.0, tk.END)
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.process_image)
        thread.daemon = True
        thread.start()
    
    def process_image(self):
        """Process the selected image"""
        try:
            image_path = self.image_path_var.get()
            output_file = self.output_file_var.get()
            
            if not output_file.endswith('.xlsx'):
                output_file += '.xlsx'
            
            # Extract text
            self.root.after(0, lambda: self.status_var.set("Extracting text from image..."))
            text = self.agent.extract_text_from_image(image_path)

            if not text:
                self.root.after(0, lambda: self.show_error("No text found in the image"))
                return

            # Extract names
            self.root.after(0, lambda: self.status_var.set("Extracting names..."))
            names = self.agent.extract_names_from_text(text)

            if not names:
                self.root.after(0, lambda: self.show_error("No names found in the text"))
                return

            # Export to Excel
            self.root.after(0, lambda: self.status_var.set("Exporting to Excel..."))
            success = self.agent.export_to_excel(names, output_file)

            if success:
                # Show results
                self.root.after(0, lambda: self.show_success(names, output_file))
            else:
                self.root.after(0, lambda: self.show_error("Failed to export file"))
                
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error occurred: {str(e)}"))
    
    def show_success(self, names, output_file):
        """Show success message and results"""
        self.progress.stop()
        self.process_button.config(state='normal')
        self.status_var.set("Operation completed successfully!")

        # Display results
        self.results_text.insert(tk.END, f"Successfully extracted {len(names)} names!\n\n")
        self.results_text.insert(tk.END, f"File saved: {output_file}\n\n")
        self.results_text.insert(tk.END, "Extracted names:\n")
        for i, name in enumerate(names, 1):
            self.results_text.insert(tk.END, f"{i}. {name}\n")

        messagebox.showinfo("Success", f"Successfully extracted {len(names)} names and saved to {output_file}")
    
    def show_error(self, message):
        """Show error message"""
        self.progress.stop()
        self.process_button.config(state='normal')
        self.status_var.set("Operation failed")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {message}")
        messagebox.showerror("Error", message)

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = NameExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()