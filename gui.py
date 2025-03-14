import tkinter as tk
from tkinter import ttk
from screen_capture import ScreenCapture
from chess_recognition import ChessRecognizer
from utils import select_screen_region, create_overlay_window

class ChessAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Assistant")
        self.chess_recognizer = ChessRecognizer()
        
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.select_button = ttk.Button(self.main_frame, text="Select Zone", command=self.select_zone)
        self.select_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.side_label = ttk.Label(self.main_frame, text="Select Side:")
        self.side_label.grid(row=1, column=0, padx=5, pady=5)
        
        self.side_var = tk.StringVar(value="white")
        self.white_radio = ttk.Radiobutton(self.main_frame, text="White", variable=self.side_var, value="white")
        self.white_radio.grid(row=1, column=1, padx=5, pady=5)
        self.black_radio = ttk.Radiobutton(self.main_frame, text="Black", variable=self.side_var, value="black")
        self.black_radio.grid(row=1, column=2, padx=5, pady=5)
        
        self.assist_button = ttk.Button(self.main_frame, text="Assist", command=self.assist)
        self.assist_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        self.quit_button = ttk.Button(self.main_frame, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.main_frame, width=400, height=300, bg="white")
        self.canvas.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        
        self.monitor = None
        self.screen_capture = None

        # For auto-assist (existing functionality)
        self.auto_assist_enabled = False
        self.auto_assist_button = ttk.Button(self.main_frame, text="Enable Auto Assist", command=self.toggle_auto_assist)
        self.auto_assist_button.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        self.auto_assist_job = None

        # New overlay attributes
        self.overlay = None
        self.overlay_canvas = None
        self.overlay_toggle_button = ttk.Button(self.main_frame, text="Show Overlay", command=self.toggle_overlay)
        self.overlay_toggle_button.grid(row=6, column=0, columnspan=3, padx=5, pady=5)
        
    def select_zone(self):
        self.monitor = select_screen_region()  # Full-screen semi-transparent selector
        print(f"Selected region: {self.monitor}")
        if self.monitor:
            self.screen_capture = ScreenCapture(self.monitor, self.chess_recognizer, self.canvas)
            print("Screen capture initialized.")
        else:
            print("No region selected.")
    
    def assist(self):
        """Manually trigger one capture-and-process cycle."""
        if not self.screen_capture:
            print("Please select a region first.")
            return
        self.screen_capture.side = self.side_var.get()
        # Draw arrows on main canvas OR update overlay if visible:
        frame = self.screen_capture.capture_and_process()
        if self.overlay_canvas:
            # Convert frame to image format and display on overlay_canvas
            # (Assuming you have code to do that, e.g., using PIL.ImageTk)
            self.update_overlay(frame)
    
    def update_overlay(self, frame):
        """Update the overlay canvas with the provided frame.
        This function should convert 'frame' (an OpenCV image) to a PhotoImage
        and then display it on self.overlay_canvas."""
        # Example using PIL:
        from PIL import Image, ImageTk
        import cv2
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.overlay_photo = ImageTk.PhotoImage(image=image)
        self.overlay_canvas.create_image(0, 0, image=self.overlay_photo, anchor=tk.NW)
    
    def toggle_auto_assist(self):
        """Toggle auto-assist mode on and off."""
        if self.auto_assist_enabled:
            if self.auto_assist_job:
                self.root.after_cancel(self.auto_assist_job)
            self.auto_assist_enabled = False
            self.auto_assist_button.config(text="Enable Auto Assist")
        else:
            self.auto_assist_enabled = True
            self.auto_assist_button.config(text="Disable Auto Assist")
            self.auto_assist()
    
    def auto_assist(self):
        """Automatically checks and processes the screen every 300ms."""
        if self.auto_assist_enabled and self.screen_capture:
            frame = self.screen_capture.capture_and_process()
            if self.overlay_canvas:
                self.update_overlay(frame)
            self.auto_assist_job = self.root.after(300, self.auto_assist)
    
    def toggle_overlay(self):
        """Toggle the overlay window that shows arrows on the selected zone."""
        if self.overlay:
            # If overlay exists, hide it
            self.overlay.destroy()
            self.overlay = None
            self.overlay_canvas = None
            self.overlay_toggle_button.config(text="Show Overlay")
        else:
            if not self.monitor:
                print("Please select a region first.")
                return
            self.overlay, self.overlay_canvas = create_overlay_window(self.monitor)
            self.overlay_toggle_button.config(text="Hide Overlay")
    
    def quit_app(self):
        """Cleans up any running tasks and quits the application."""
        if self.auto_assist_job:
            self.root.after_cancel(self.auto_assist_job)
        if self.screen_capture and self.screen_capture.engine:
            try:
                self.screen_capture.engine.quit()
            except Exception as e:
                print("Error quitting engine:", e)
        if self.overlay:
            self.overlay.destroy()
        print("Closing application...")
        self.root.quit()
        self.root.destroy()
