import tkinter as tk
from tkinter import ttk
from screen_capture import ScreenCapture
from chess_recognition import ChessRecognizer
from utils import select_screen_region


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

        self.quit_button = ttk.Button(self.main_frame, text="Quit", command=root.quit)
        self.quit_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.canvas = tk.Canvas(self.main_frame, width=400, height=300, bg="white")
        self.canvas.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

        self.monitor = None
        self.screen_capture = None

    def select_zone(self):
        self.monitor = select_screen_region()
        self.screen_capture = ScreenCapture(self.monitor, self.chess_recognizer, self.canvas)

    def assist(self):
        if self.screen_capture:
            self.screen_capture.side = self.side_var.get()
            self.screen_capture.capture_and_process()
        else:
            print("Please select a zone first.")

