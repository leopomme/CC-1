import cv2
import numpy as np
import mss
from PIL import Image, ImageTk
from utils import has_significant_change, draw_arrows_on_frame
import chess
import chess.engine
import tkinter as tk


class ScreenCapture:
    def __init__(self, monitor, chess_recognizer, canvas):
        self.monitor = monitor
        self.chess_recognizer = chess_recognizer
        self.canvas = canvas
        self.sct = mss.mss()

    def capture_screen(self):
        print("Capturing screen")
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        return frame

    def capture_and_process(self):
        print("Capturing and processing frame")
        frame = self.capture_screen()
        self.process_frame(frame)

    def process_frame(self, frame):
        print("Processing frame")
        results, grid = self.chess_recognizer._detect_chessboard(frame)
        if results:
            print("Chessboard detected")
            board = self.chess_recognizer.setup_board(results)
            stockfish_path = '/usr/local/bin/stockfish'
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                best_moves_current = self.get_best_moves(engine, board)
                board.push(chess.Move.null())
                best_moves_other = self.get_best_moves(engine, board)
                board.pop()
                frame = draw_arrows_on_frame(frame, best_moves_current, best_moves_other, board, grid)
            self.display_frame(frame)
        else:
            print("No chessboard detected")

    def get_best_moves(self, engine, board):
        limit = chess.engine.Limit(time=2.0)
        info = engine.analyse(board, limit, multipv=5)
        moves = [info[i]['pv'][0] for i in range(len(info))]
        return moves

    def display_frame(self, frame):
        print("Displaying frame")
        frame_resized = cv2.resize(frame, (400, 300))  # Adjust to canvas size
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
