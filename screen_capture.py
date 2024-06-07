import cv2
import numpy as np
import mss
from PIL import Image, ImageTk
from utils import has_significant_change, draw_arrows_on_frame
import chess
import chess.engine
import tkinter as tk
import logging

logging.basicConfig(level=logging.INFO)

class ScreenCapture:
    def __init__(self, monitor, chess_recognizer, canvas):
        self.monitor = monitor
        self.chess_recognizer = chess_recognizer
        self.canvas = canvas
        self.sct = mss.mss()
        self.side = "white"
        self.stockfish_path = '/usr/local/bin/stockfish'
        self.engine = None  

    def capture_screen(self):
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        return frame

    def capture_and_process(self):
        frame = self.capture_screen()
        self.process_frame(frame)

    def process_frame(self, frame):
        results, grid = self.chess_recognizer._detect_chessboard(frame)
        if results:
            board = self.chess_recognizer.setup_board(results, self.side)
            print(board)
            self.ensure_engine_running()
            try:
                best_moves_current = self.get_best_moves(self.engine, board)
                board.push(chess.Move.null())
                best_moves_other = self.get_best_moves(self.engine, board)
                board.pop()
                frame = draw_arrows_on_frame(frame, best_moves_current, best_moves_other, board, grid, self.side)
                self.display_frame(frame)
            except Exception as e:
                logging.error(f"Unexpected error during move analysis: {e}")
        else:
            logging.info("No chessboard detected")

    def ensure_engine_running(self):
        try:
            if self.engine is None:
                logging.info("Starting Stockfish engine")
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            else:
                # Send a command to check if the engine is still responsive
                self.engine.ping()
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
            logging.error(f"Engine error detected: {e}. Restarting engine.")
            self.engine.quit()
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def get_best_moves(self, engine, board):
        try:
            limit = chess.engine.Limit(time=2.0, depth=10)
            info = engine.analyse(board, limit, multipv=5)
            moves = []
            for i in range(len(info)):
                if 'pv' in info[i]:
                    moves.append(info[i]['pv'][0])
                else:
                    logging.error(f"'pv' key not found in info[{i}]: {info[i]}")
            return moves
        except chess.engine.EngineTerminatedError as e:
            logging.error(f"Engine process died unexpectedly: {e}")
            self.restart_engine()
            return []
        except Exception as e:
            logging.error(f"Unexpected error during analyse: {e}")
            try:
                result = engine.play(board, chess.engine.Limit(time=2.0))
                if result.move:
                    return [result.move]
            except chess.engine.EngineTerminatedError as e:
                logging.error(f"Engine process died unexpectedly during fallback to play: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during fallback to play: {e}")
            return []

    def restart_engine(self):
        logging.info("Restarting Stockfish engine")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def display_frame(self, frame):
        frame_resized = cv2.resize(frame, (400, 300))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

