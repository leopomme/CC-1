import torch
import cv2
from model import ChessPieceCNN
import numpy as np
import chess
import chess.engine
import tkinter as tk

def load_model(model_path):
    print("Loading model")
    model = ChessPieceCNN(num_classes=13)  # Adjust according to your model definition
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def has_significant_change(frame1, frame2, threshold=30):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresh)
    print(f"Non-zero pixel count: {non_zero_count}")
    return non_zero_count > threshold

def select_screen_region():
    import tkinter as tk
    root = tk.Tk()
    selector = ScreenRegionSelector(root)
    root.mainloop()
    return selector.rect_coords

def draw_arrows_on_frame(frame, best_moves_current, best_moves_other, board, grid, side):
    def get_square_center(rank, file):
        if side == "black":
            rank = 7 - rank
            file = 7 - file
        top_left = grid[7 - rank][file]
        bottom_right = grid[7 - rank + 1][file + 1]
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        return center_x, center_y

    for i, move in enumerate(best_moves_current):
        from_square = move.from_square
        to_square = move.to_square
        from_center = get_square_center(chess.square_rank(from_square), chess.square_file(from_square))
        to_center = get_square_center(chess.square_rank(to_square), chess.square_file(to_square))
        color = (0, 255, 0)
        thickness = max(15 - 3*i, 1)
        cv2.arrowedLine(frame, from_center, to_center, color, thickness, tipLength=0.3)

    for i, move in enumerate(best_moves_other):
        from_square = move.from_square
        to_square = move.to_square
        from_center = get_square_center(chess.square_rank(from_square), chess.square_file(from_square))
        to_center = get_square_center(chess.square_rank(to_square), chess.square_file(to_square))
        color = (0, 0, 255)
        thickness = max(15 - 3*i, 1)
        cv2.arrowedLine(frame, from_center, to_center, color, thickness, tipLength=0.3)

    return frame


class ScreenRegionSelector:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.3)
        self.root.bind("<Escape>", self.quit)
        
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rect_coords = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
    
    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)
    
    def on_move_press(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
    
    def on_button_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        self.rect_coords = (int(self.start_x), int(self.start_y), int(end_x), int(end_y))
        self.quit()
    
    def quit(self, event=None):
        self.root.destroy()
