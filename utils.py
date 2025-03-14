import torch
import cv2
from model import ChessPieceCNN
import numpy as np
import chess
import chess.engine
import tkinter as tk
import math

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

import math
import cv2
import numpy as np
import chess

def draw_arrows_on_frame(frame, best_moves_current, best_moves_other, board, grid, side, alpha=0.6):
    """
    Draws arrows for the top three moves for both players.
    Green arrows indicate your moves and red arrows indicate your opponent's moves.
    Arrow thickness and arrowhead size are scaled dynamically.
    """
    overlay = frame.copy()

    def get_square_center(rank, file):
        # Adjust coordinate mapping based on selected side
        if side == "black":
            rank = 7 - rank
            file = 7 - file
        top_left = grid[7 - rank][file]
        bottom_right = grid[7 - rank + 1][file + 1]
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        return center_x, center_y

    # Draw current player's moves (green)
    for i, move in enumerate(best_moves_current[:3]):
        from_square = move.from_square
        to_square = move.to_square
        from_center = get_square_center(chess.square_rank(from_square), chess.square_file(from_square))
        to_center = get_square_center(chess.square_rank(to_square), chess.square_file(to_square))
        thickness = max(10 - i * 3, 3)  # Best move has the thickest line
        if i == 0:
            color = (0, 255, 0)       # Bright green
        elif i == 1:
            color = (50, 205, 50)     # Medium green
        else:
            color = (100, 255, 100)   # Lighter green
        overlay = draw_custom_arrowhead(overlay, from_center, to_center, color, thickness)

    # Draw opponent's moves (red)
    for i, move in enumerate(best_moves_other[:3]):
        from_square = move.from_square
        to_square = move.to_square
        from_center = get_square_center(chess.square_rank(from_square), chess.square_file(from_square))
        to_center = get_square_center(chess.square_rank(to_square), chess.square_file(to_square))
        thickness = max(10 - i * 3, 3)
        if i == 0:
            color = (0, 0, 255)       # Bright red
        elif i == 1:
            color = (50, 50, 205)     # Medium red
        else:
            color = (100, 100, 255)   # Lighter red
        overlay = draw_custom_arrowhead(overlay, from_center, to_center, color, thickness)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

import math
import cv2
import numpy as np

def draw_custom_arrowhead(image, from_pt, to_pt, color, thickness):
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return image  # Avoid division by zero

    ux = dx / length
    uy = dy / length

    # Scale arrowhead size relative to thickness:
    arrow_head_length = max(20, thickness * 2)
    arrow_head_width = max(10, thickness * 1.5)

    # Calculate the base of the arrowhead
    base_x = to_pt[0] - arrow_head_length * ux
    base_y = to_pt[1] - arrow_head_length * uy

    # Perpendicular vector for arrowhead width
    perp_x = -uy
    perp_y = ux

    left_x = base_x + (arrow_head_width / 2.0) * perp_x
    left_y = base_y + (arrow_head_width / 2.0) * perp_y
    right_x = base_x - (arrow_head_width / 2.0) * perp_x
    right_y = base_y - (arrow_head_width / 2.0) * perp_y

    arrow_points = np.array([
        [int(to_pt[0]), int(to_pt[1])],
        [int(left_x), int(left_y)],
        [int(right_x), int(right_y)]
    ], dtype=np.int32)

    cv2.fillPoly(image, [arrow_points], color)
    cv2.line(image, from_pt, to_pt, color, thickness, lineType=cv2.LINE_AA)
    return image


def select_screen_region():
    root = tk.Tk()
    # Set the window to full screen and semi-transparent
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.3)
    root.configure(background='gray')
    
    rect_coords = [None]  # Will hold the final (x1, y1, x2, y2)
    start_x = [None]
    start_y = [None]
    rect = [None]
    
    canvas = tk.Canvas(root, bg="gray")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    def on_button_press(event):
        start_x[0] = canvas.canvasx(event.x)
        start_y[0] = canvas.canvasy(event.y)
        if rect[0]:
            canvas.delete(rect[0])
        rect[0] = canvas.create_rectangle(start_x[0], start_y[0],
                                          start_x[0], start_y[0],
                                          outline='red', width=2)
        print("Button pressed at:", start_x[0], start_y[0])
    
    def on_move_press(event):
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)
        if rect[0]:
            canvas.coords(rect[0], start_x[0], start_y[0], cur_x, cur_y)
    
    def on_button_release(event):
        end_x = canvas.canvasx(event.x)
        end_y = canvas.canvasy(event.y)
        rect_coords[0] = (int(start_x[0]), int(start_y[0]), int(end_x), int(end_y))
        print("Button released at:", end_x, end_y)
        print("Region selected:", rect_coords[0])
        # Quit and then destroy the window
        root.quit()
    
    # Bind the mouse events to the canvas...
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    # Also bind the button release to the root (in case the event escapes the canvas)
    root.bind("<ButtonRelease-1>", on_button_release)
    
    root.mainloop()
    root.destroy()  # Ensure the window is fully closed after mainloop exits
    return rect_coords[0]

def create_overlay_window(monitor):
    """Creates a borderless overlay window that covers the selected region
    and makes its background transparent so only the drawn arrows are visible."""
    overlay = tk.Toplevel()
    overlay.overrideredirect(True)  # Remove window borders
    overlay.attributes("-topmost", True)
    # Position the overlay at the same coordinates as the selected region
    x, y, x2, y2 = monitor
    width = x2 - x
    height = y2 - y
    overlay.geometry(f"{width}x{height}+{x}+{y}")
    
    # Set the background to a color that we make transparent (e.g., white)
    overlay.configure(bg='white')
    overlay.attributes("-transparentcolor", "white")
    
    # Create a canvas for drawing
    canvas = tk.Canvas(overlay, width=width, height=height, highlightthickness=0, bg="white")
    canvas.pack()
    return overlay, canvas
