import numpy as np
import random
import game_environment as g
import torch
from torch import nn
import math

def to_coord(index, width):
    row = index // width
    col = index % width
    return row, col

def prob_bomb(probabilities, width, threshold=0.99):
    high_confidence_indices = np.where(probabilities > threshold)[0]  # Get indices of values above the threshold
    return to_coord(high_confidence_indices, width)

def get_best_move(probabilities, width):
    min_index = np.argmin(probabilities) 
    max_index = np.argmax(probabilities)
    best = np.max(math.abs(1 - min_index), math.abs(max_index - 1))
    return to_coord(best, width)

def delete_arr(list, array):
    for i, arr in enumerate(list):
        if np.array_equal(arr, array):
            list.pop(i)
            return

def dict_to_2d(dict, width, height):
    array = np.full((height, width), fill_value=np.nan)
    for (x, y), value in dict.items():
        array[y, x] = value
    return array

def twoD_to_1d(twod_arr):
    lst = []
    for row in twod_arr:
        for item in row:
            lst.append(item)
    return np.array(lst)

def create_reveal_mask(board):
    height, width = board.shape
    mask = np.zeros_like(board, dtype=int)
    revealed = board != 9
    for y in range(height):
        for x in range(width):
            if revealed[y, x]:
                mask[y, x] = 1
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            mask[ny, nx] = 1
    return mask

def compare_flags(marked_mines, actual_mines):
    correct = np.sum((marked_mines == 10) & (actual_mines == 1))
    incorrect = np.sum((marked_mines == 10) & (actual_mines == 0))
    return correct, incorrect


def logic(model, width = 9, height = 9, mines = 10):
    model.eval()
    
    game = g.MinesweeperGame(width, height, mines)
    cells_remaining = []
    #inferred_safe = []
    inferred_mine = []
    priv_board_values = {}
    #remain_cells = np.zeros([height * width, 2])
    for i in range(height):
        for j in range(width):
            cells_remaining.append(np.array([j,i]))
            priv_board_values[j,i] = 9
    first_x = random.randint(0,width-1)
    first_y = random.randint(0,height-1)

    #game.print_board()
    #print("awooooooooga")
    #initial starting values

    game.initialize_mines(first_x, first_y)
    first_index = first_x,first_y
    priv_board_values[first_index] = game.reveal_cell(first_index[0], first_index[1])
    delete_arr(cells_remaining, first_index)

    revealed_cells = 1
    bombs_flagged = 0
    while not game.game_over:
        twod_priv_board = dict_to_2d(priv_board_values, width, height)
        computed_array = model(twod_priv_board)
        #game.print_board()
        #for the real code, pass the board thru the model to obtain the computed array of probabilities
        index_coord = get_best_move(computed_array, width)
        if computed_array[index_coord] > 0.5:
            priv_board_values[index_coord] = game.reveal_cell(index_coord[0], index_coord[1])
            revealed_cells += 1
        else:
            game.flag_cell(index_coord[0], index_coord[1])
            inferred_mine.append()
            bombs_flagged += 1
        delete_arr(cells_remaining, index_coord)
        #print(priv_board_values)
    
    #final board as 2d arr
    twod_priv_board = dict_to_2d(priv_board_values, width, height)
    #final board as 1d arr
    oned_priv_board = twoD_to_1d(twod_priv_board)
    #unhidden board
    unhidden_board_2d = game.get_board()
    #unhidden board but only mines
    unhidden_mines_2d = (unhidden_board_2d == -1).astype(int)
    #1s and 0s for border cells
    masked_hidden_board_2d = create_reveal_mask(twod_priv_board)
    #unhidden mines within border cells
    mask_unhidden_mines_2d = ((masked_hidden_board_2d == 1) & ((unhidden_mines_2d) == 1)).astype(int)
    #unhidden mines within border cells 1D
    mask_unhidden_mines_1d = twoD_to_1d(mask_unhidden_mines_2d)

    if revealed_cells == width * height - mines:
        win = 1
    else:
        win = 0
        
    correct_flags, incorrect_flags = compare_flags(twod_priv_board, unhidden_mines_2d)
    return win, revealed_cells, correct_flags, incorrect_flags
    
    
def train_cnn (alpha, model, epochs, convergence, width, height):
    cnn_optimizer = optim.Adam(model.parameters(), lr = alpha )
    loss_function = torch.nn.CrossEntropyLoss()
    in_progress = False
    game = None
    priv_board_values = None
    losses = []
    prev_loss = math.inf
    for epochs in range(epochs):
        total_loss = 0
        model.train()
        isOver, game, twod_priv_board, priv_board_values, mask_unhidden_mines_1d = logic(model, game, priv_board_values, in_progress)
        if(isOver):
            in_progress = False
        else:
            in_progress = True
        cnn_optimizer.zero_grad()
        input = torch.from_numpy(twod_priv_board).float()
        output = torch.from_numpy(mask_unhidden_mines_1d).float()
        input = input.unsqueeze(0).unsqueeze(0)        
        logits = model( input )
        print(logits)
        loss = loss_function( logits, output )

        loss.backward()
        cnn_optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (width * height)
        losses.append([avg_loss])
        if epochs > 0 and abs(prev_loss - avg_loss) < convergence:
            print(f"Convergence reached after {epochs+1} epochs.")
            break
        print(avg_loss)
    return losses
    
    '''
    import numpy as np
import random
import game_environment as g
import torch
from torch import nn
import math

def to_coord(index, width):
    row = index // width
    col = index % width
    return row, col

def prob_bomb(probabilities, width, threshold=0.99):
    high_confidence_indices = np.where(probabilities > threshold)[0]  # Get indices of values above the threshold
    return to_coord(high_confidence_indices, width)

def get_best_move(probabilities, cells_remaining):
    cells_remaining = np.concatenate(cells_remaining).astype(int)
    remaining_indices = torch.tensor(cells_remaining, dtype=torch.long)
    filtered_probs = probabilities[remaining_indices]

    max_prob, max_rel_index = torch.max(filtered_probs, 0)
    min_prob, min_rel_index = torch.min(filtered_probs, 0)

    max_index = remaining_indices[max_rel_index]
    min_index = remaining_indices[min_rel_index]

    max_prob = max_prob.item()
    min_prob = min_prob.item()
    #closeness_max = 1 - max_prob
    #closeness_min = min_prob

    #hyperparameter of threshold for flagging
    if max_prob < 0.9:
        return min_index.item(), max_index.item()
    else:
        return min_index.item(), -1

def delete_arr(list, array):
    for i, arr in enumerate(list):
        if np.array_equal(arr, array):
            list.pop(i)
            return

def dict_to_2d(dict, width, height):
    array = np.full((height, width), fill_value=np.nan)
    for (x, y), value in dict.items():
        array[y, x] = value
    return array

def twoD_to_1d(twod_arr):
    lst = []
    for row in twod_arr:
        for item in row:
            lst.append(item)
    return np.array(lst)

def create_reveal_mask(board):
    height, width = board.shape
    mask = np.zeros_like(board, dtype=int)
    revealed = board != 9
    for y in range(height):
        for x in range(width):
            if revealed[y, x]:
                mask[y, x] = 1
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            mask[ny, nx] = 1
    return mask

def compare_flags(marked_mines, actual_mines):
    correct = np.sum((marked_mines == 10) & (actual_mines == 1))
    incorrect = np.sum((marked_mines == 10) & (actual_mines == 0))
    return correct, incorrect

def init_game(width, height, mines):
    game = g.MinesweeperGame(width, height, mines)
    cells_remaining = []
    #inferred_safe = []
    inferred_mine = []
    priv_board_values = {}
    #remain_cells = np.zeros([height * width, 2])
    for i in range(height):
        for j in range(width):
            cells_remaining.append(np.array([j,i]))
            priv_board_values[j,i] = 9
    first_x = random.randint(0,width-1)
    first_y = random.randint(0,height-1)

    #game.print_board()
    #print("awooooooooga")
    #initial starting values

    game.initialize_mines(first_x, first_y)
    first_index = first_x,first_y
    priv_board_values[first_index] = game.reveal_cell(first_index[0], first_index[1])
    delete_arr(cells_remaining, first_index)
    return game, cells_remaining, priv_board_values
    

def logic(model, game, priv_board_values, inProgress, cells_remaining, width = 9, height = 9, mines = 10):
    model.eval()
    if (not inProgress):
        game, cells_remaining, priv_board_values = init_game(width, height, mines)
        twod_priv_board = dict_to_2d(priv_board_values, width, height)
        unhidden_board_2d = game.get_board()
        #print(twod_priv_board)
        unhidden_mines_2d = (unhidden_board_2d == -1).astype(int)
        masked_hidden_board_2d = create_reveal_mask(twod_priv_board)
        mask_unhidden_mines_2d = ((masked_hidden_board_2d == 1) & ((unhidden_mines_2d) == 1)).astype(int)
        mask_unhidden_mines_1d = twoD_to_1d(mask_unhidden_mines_2d)
        win = game.check_win()
        return False, game, twod_priv_board, priv_board_values, mask_unhidden_mines_1d, win, cells_remaining
        #inferred_mine = []
    #else:
        #cells_remaining = game.get_remaining_cells()
        #inferred_mine = game.get_flags()
        
    revealed_cells = 1
    bombs_flagged = 0
    
    twod_priv_board = dict_to_2d(priv_board_values, width, height)
    
    tensor_input = torch.from_numpy(twod_priv_board).float()  # Ensuring the data type is float
    tensor_input = tensor_input.unsqueeze(0)
    computed_array = model(tensor_input)
    #game.print_board()
    #for the real code, pass the board thru the model to obtain the computed array of probabilities
    index, flag_i = get_best_move(computed_array, cells_remaining)
    index_coord = to_coord(index, width)
    flag_coord = to_coord(flag_i, width)

    if computed_array[index] > 0.5:
        priv_board_values[index_coord] = game.reveal_cell(index_coord[0], index_coord[1])
        revealed_cells += 1
        delete_arr(cells_remaining, index_coord)
        print("awoog")
    else:
        game.flag_cell(index_coord[0], index_coord[1])
        #inferred_mine.append(index_coord)
        bombs_flagged += 1
        #print(priv_board_values)
    
    isOver = game.is_game_over()
    print(twod_priv_board)
    win = game.check_win()
    print(cells_remaining)
    #print(win)
    #final board as 2d arr
    twod_priv_board = dict_to_2d(priv_board_values, width, height)
    #final board as 1d arr
    oned_priv_board = twoD_to_1d(twod_priv_board)
    #unhidden board
    unhidden_board_2d = game.get_board()
    #unhidden board but only mines
    unhidden_mines_2d = (unhidden_board_2d == -1).astype(int)
    #1s and 0s for border cells
    masked_hidden_board_2d = create_reveal_mask(twod_priv_board)
    #unhidden mines within border cells
    mask_unhidden_mines_2d = ((masked_hidden_board_2d == 1) & ((unhidden_mines_2d) == 1)).astype(int)
    #unhidden mines within border cells 1D
    mask_unhidden_mines_1d = twoD_to_1d(mask_unhidden_mines_2d)
        
    correct_flags, incorrect_flags = compare_flags(twod_priv_board, unhidden_mines_2d)
    return isOver, game, twod_priv_board, priv_board_values, mask_unhidden_mines_1d, win, cells_remaining
    '''