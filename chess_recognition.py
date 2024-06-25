import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import chess
import chess.engine
from utils import load_model
import numpy as np

class ChessRecognizer:
    def __init__(self, model_path='best_model.pth'):
        self.model = load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.class_names = ['black_bishop', 'black_king', 'black_queen', 'black_knight', 'black_rook',
                            'black_pawn', 'white_bishop', 'white_king', 'white_queen', 'white_knight',
                            'white_rook', 'white_pawn', 'empty']
        self.grid = None  # Cache the grid coordinates

    def classify_chessboard(self, image_path):
        image = cv2.imread(image_path)
        results, grid = self._detect_chessboard(image)
        return results, grid

    def _detect_chessboard(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 900)

        if lines is not None and self.grid is None:
            intersections = self._find_intersections(lines, frame.shape)
            if len(intersections) >= 64:
                self.grid = self._sort_intersections(intersections)
        
        if self.grid is not None:
            results = self._classify_pieces(frame, self.grid)
            return results, self.grid
        return {}, []

    def _find_intersections(self, lines, shape):
        intersections = []
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            for rho2, theta2 in lines[:, 0]:
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                x1 = a2 * rho2
                y1 = b2 * rho2
                denom = a * b2 - a2 * b
                if denom != 0:
                    x = (b2 * x0 - b * x1) / denom
                    y = (a * y1 - a2 * y0) / denom
                    if 0 <= x < shape[1] and 0 <= y < shape[0]:
                        intersections.append((int(x), int(y)))
        return sorted(set(intersections))

    def _sort_intersections(self, intersections):
        intersections.sort(key=lambda x: (x[1], x[0]))  # Sort by y, then by x
        grid = []
        grid_size = 9  # 8x8 board has 9 lines
        for i in range(0, len(intersections), grid_size):
            row = sorted(intersections[i:i + grid_size], key=lambda x: x[0])
            grid.append(row)
        return grid

    def _classify_pieces(self, frame, grid):
        results = {}
        for i in range(8):
            for j in range(8):
                top_left = grid[i][j]
                top_right = grid[i][j + 1]
                bottom_left = grid[i + 1][j]
                bottom_right = grid[i + 1][j + 1]

                # Extract the cell image
                cell_image = frame[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]
                cell_image_rgb = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)  # Ensure 3 channels
                cell_image_pil = TF.to_pil_image(cell_image_rgb)
                image_tensor = self.transform(cell_image_pil).unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1).squeeze(0)
                    results[(i, j)] = probabilities

        return results

    def setup_board(self, results, side):
        board = chess.Board()
        board.clear()

        piece_map = {
            'black_bishop': chess.BISHOP, 'black_king': chess.KING, 'black_queen': chess.QUEEN,
            'black_knight': chess.KNIGHT, 'black_rook': chess.ROOK, 'black_pawn': chess.PAWN,
            'white_bishop': chess.BISHOP, 'white_king': chess.KING, 'white_queen': chess.QUEEN,
            'white_knight': chess.KNIGHT, 'white_rook': chess.ROOK, 'white_pawn': chess.PAWN
        }
        color_map = {piece: (chess.BLACK if 'black' in piece else chess.WHITE) for piece in piece_map.keys()}

        for coord, probs in results.items():
            max_prob_index = torch.argmax(probs).item()
            max_prob_piece = self.class_names[max_prob_index]
            if max_prob_piece != 'empty':
                piece = piece_map[max_prob_piece]
                color = color_map[max_prob_piece]
                rank = 7 - coord[0] if side == "white" else coord[0]
                file = coord[1] if side == "white" else 7 - coord[1]
                square = chess.square(file, rank)
                board.set_piece_at(square, chess.Piece(piece, color))

        return board
