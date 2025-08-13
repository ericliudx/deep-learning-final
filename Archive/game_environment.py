import numpy as np
import random

class MinesweeperGame:
    def __init__(self, width, height, num_mines):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = np.zeros((height, width), dtype=int)
        self.revealed = np.full((height, width), False, dtype=bool)
        self.mines = set()
        self.flags = np.full((height, width), False, dtype=bool)  # To track flags
        self.game_over = False
        self.cells_remaining = np.full((height, width), True, dtype=bool)
        self.calculate_clues()

    def initialize_mines(self, first_x, first_y):
        mines_count = 0
        while mines_count < self.num_mines:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) != (first_x, first_y) and (x, y) not in self.mines:
                self.mines.add((x, y))
                self.board[y, x] = -1 
                mines_count += 1
        self.calculate_clues()

    def calculate_clues(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] == -1:
                    continue
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.board[ny, nx] == -1:
                                count += 1
                self.board[y, x] = count

    def reveal_cell(self, x, y):
        if self.revealed[y, x]:
            return self.board[y, x]
        
        self.revealed[y, x] = True
        self.cells_remaining[y, x] = False
        
        if (x, y) in self.mines:
            self.game_over = True
            return -1
        return self.board[y, x]
    
    def flag_cell(self, x, y):
        if not self.revealed[y, x]:
            self.flags[y, x] = True
            self.board[y, x] = 10 

    def unflag_cell(self, x, y):
        if self.flags[y, x]:
            self.flags[y, x] = False
            self.board[y, x] = 0 
    
    def is_game_over(self):
        if np.any(self.revealed & (self.board == -1)):
            self.game_over = True
            return True
        return self.check_win()

    def print_board(self):
        for y in reversed(range(self.height)):
            row = ""
            for x in range(self.width):
                if not self.revealed[y, x]:
                    row += '9  '
                else:
                    row += str(self.board[y, x]) + '  '
            print(row)
            
    def get_board(self):
        return self.board

    def check_win(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] != -1 and not self.revealed[y, x]:
                    return False
        return True
    
    def get_remaining_cells(self):
        return self.cells_remaining
    def get_flags(self):
        return self.flags
    