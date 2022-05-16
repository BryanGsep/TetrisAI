import gym
import pygame
import numpy as np
import random

# Global variables
pygame.font.init()
s_width = 800
s_height = 700
play_width = 300  # meaning 300//10 = 30 width per block
play_height = 600  # meaning 600//20 = 30 height per block
block_size = 30
top_left_x = (s_width - play_width) // 2
top_left_y = (s_height - play_height)

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '...0.',
      '...0.',
      '..00.',
      '.....'],
     ['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.000.',
      '...0.',
      '.....',
      '.....']]

L = [['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '...0.',
      '.000.',
      '.....',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]



shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0)]

# index 0-6 represent shape

## Define game piece
"""-------------------------------------------------------------------"""

class Piece(object):
    rows = 20  # height
    columns = 10  # width

    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.shape_index = shapes.index(self.shape)
        self.color = shape_colors[self.shape_index]
        self.rotation = 0  # number from 0-3


## End of game piece
"""-------------------------------------------------------------------"""

## Game running function
"""-------------------------------------------------------------------"""

def create_grid(locked_positions):
    state = [[(0, 0, 0) for x in range(10)] for y in range(20)]
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                state[i][j] = c
    return state

def convert_shape_format(piece):
    piece_pos = []
    ## Determine rotation direction of current shape
    format = piece.shape[piece.rotation % len(piece.shape)]
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                piece_pos.append((piece.x + j - 2, piece.y + i - 4))
    return piece_pos

def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub]
    piece_pos = convert_shape_format(shape)
    for pos in piece_pos:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
            elif pos[0] > 9 or pos[0] < 0:
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False

def get_shape():
    global shapes, shape_colors
    return Piece(5, 0, random.choice(shapes))


def get_col_height(state, col):
    state_height, state_width, _ = np.array(state).shape
    col_height = 0
    if col <-1 or col > state_width:
        return IndexError('Column index should in the range from 0 to {}'.format(state_width))
    else:
        for i in range(state_height):
            if state[i][col] != (0,0,0):
                col_height = state_height-i-1
                break
    return col_height

#def calculate_terrain(state):
#    """ Total different between adjacent column"""
#    state_height, state_width, _ = np.array(state).shape
#    terrain = 0
#    curr_height = 0
#    next_height = 0
#    for i in range(state_width):
#        if i == 0:
#            curr_height = get_col_height(state, i)
#        else:
#            next_height = get_col_height(state, i)
#            terrain += np.abs(next_height-curr_height)
#            curr_height = next_height
#    return terrain

def calculate_hole_nb(state):
    """ Calculate total holes (can't fill point) in current structure)"""
    state_height, state_width, _ = np.array(state).shape
    nb_hole = 0
    for j in range(state_width):
        start_counting = False
        for i in range(state_height):
            if state[i][j]!=(0,0,0) and not start_counting:
                start_counting = True
            elif state[i][j] == (0,0,0) and start_counting:
                nb_hole +=1
    return nb_hole

#def get_max_height(state):
#    state_height, state_width, _ = np.array(state).shape
#    return max(get_col_height(state, i) for i in range(state_width))

#def get_min_height(state):
#    state_height, state_width, _ = np.array(state).shape
#    return min(get_col_height(state, i) for i in range(state_width))

def get_height_vs_base(state):
    state_height,state_width, _ = np.array(state).shape
    min = 20
    heights = []
    for i in range(state_width):
        height = get_col_height(state, i)
        heights.append(height)
        if min > height:
            min = height
    for i in range(state_width):
        heights[i] = heights[i]-min
    return heights


def clear_rows(state, locked_positions):
    # need to see if row is clear the shift every other row above down one
    ind = 0  # Position of the last cleared row
    inc = 0  # Determine number of cleared row
    for i in range(len(state) - 1, -1, -1):
        row = state[i]
        if (0, 0, 0) not in row:
            inc += 1
            # add positions to remove from locked
            ind = i
            for j in range(len(row)):
                try:
                    del locked_positions[(j, i)]
                except:
                    continue
            break
    if inc > 0:
        for key in sorted(list(locked_positions), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked_positions[newKey] = locked_positions.pop(key)
        return True
    return False

## End of game running function
"""-------------------------------------------------------------------"""

## Drawing function
"""-------------------------------------------------------------------"""

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, True, color)
    surface.blit(label, (top_left_x + play_width / 2- (label.get_width() / 2),
                         top_left_y + play_height / 3 - label.get_height() / 2))


def draw_grid(surface, row, col):
    sx = top_left_x
    sy = top_left_y
    for i in range(row):
        pygame.draw.line(surface, (128, 128, 128), (sx, sy + i * 30), (sx + play_width, sy + i * 30))  # horizontal line
        for j in range(col):
            pygame.draw.line(surface, (128, 128, 128), (sx + j * 30, sy),
                             (sx + j * 30, sy * play_height))  # vertical line


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Hinh tiep theo', True, (255, 255, 255))
    sx = top_left_x + play_width + 20
    sy = top_left_y + play_height / 2 - 200
    format = shape.shape[shape.rotation % len(shape.shape)]
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j * 30, sy + i * 30, 30, 30), 0)
    surface.blit(label, (sx + 10, sy - 30))


def draw_score(score, surface):
    font = pygame.font.SysFont('comicsans', 40)
    font2 = pygame.font.SysFont('comicsans', 50)
    label1 = font.render('Score', True, (255, 255, 0))
    label2 = font2.render(str(score), True, (255, 255, 255))
    surface.blit(label1, (top_left_x - 180, top_left_y + play_height / 2 + 30))
    surface.blit(label2, (top_left_x - 150, top_left_y + play_height / 2 + 80))


def draw_window(surface, grid):
    surface.fill((0, 0, 0))
    # Tetris Title
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('TRO CHOI XEP HINH', True, (255, 255, 255))
    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 10))
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j * 30, top_left_y + i * 30, 30, 30), 0)
    # draw grid and border
    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 5)
    # pygame.display.update()

## End of drawing function
"""-------------------------------------------------------------------"""

class TetrisEnv(gym.Env):
    def __init__(self):
        self.locked_positions = {}
        self.change_piece = False
        self.curr_piece = get_shape()
        self.next_piece = get_shape()
        # Actions: we can move left, right, down, space down and rotate
        self.action_space = gym.spaces.Discrete(40)
        # Grid matrix
        self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                                                high=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,20,20,20,20,20,20,20,20,20,20]))
        self.observation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.observation[self.curr_piece.shape_index] = 1
        self.observation[self.next_piece.shape_index+7] = 1
        # Set start space
        self.state = create_grid(self.locked_positions)
        self.state_memo = [self.state]
        # Set number of piece length
        self.score = 0
        self.position_reward = 0
        self.set_win = False
        self.nb_holes = 0

    def move_left(self):
        self.curr_piece.x -= 1
        if not valid_space(self.curr_piece, self.state):
            self.curr_piece.x += 1
        self.add_piece_into_state()
        self.state_memo.append(self.state)

    def move_right(self):
        self.curr_piece.x += 1
        if not valid_space(self.curr_piece, self.state):
            self.curr_piece.x -= 1
        self.add_piece_into_state()
        self.state_memo.append(self.state)

    def rotate(self):
        self.curr_piece.rotation = (self.curr_piece.rotation + 1) % len(self.curr_piece.shape)
        if not valid_space(self.curr_piece, self.state):
            self.curr_piece.rotation = (self.curr_piece.rotation - 1) % len(self.curr_piece.shape)
        self.add_piece_into_state()
        self.state_memo.append(self.state)

    def move_down(self):
        self.curr_piece.y += 1
        if not valid_space(self.curr_piece, self.state):
            self.curr_piece.y -= 1
            self.change_piece = True
        self.add_piece_into_state()
        self.state_memo.append(self.state)

    def fall_down(self):
        while valid_space(self.curr_piece, self.state):
            self.curr_piece.y += 1
        self.curr_piece.y -= 1
        self.change_piece = True
        self.add_piece_into_state()
        self.state_memo.append(self.state)

    def add_piece_into_state(self):
        shape_pos = convert_shape_format(self.curr_piece)
        # add piece to the state
        self.state = create_grid(self.locked_positions)
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                self.state[y][x] = self.curr_piece.color


    def step(self, action):
        reward = 0
        self.state_memo = [self.state]
        done = False
        info = ""

        if action >= 0 and action <= 9:
            if action >= 0 and action <=4:
                for _ in range(5-action):
                    self.move_left()
            elif action >= 6 and action <=9:
                for _ in range(action-5):
                    self.move_right()
            self.fall_down()

        elif action >= 10 and action <= 19:
            self.rotate()
            if action >= 10 and action <= 14:
                for _ in range(15-action):
                    self.move_left()
            elif action >= 16 and action <= 19:
                for _ in range(action-15):
                    self.move_right()
            self.fall_down()

        elif action >= 20 and action <= 29:
            self.rotate()
            self.rotate()
            if action >= 20 and action <= 24:
                for _ in range(25-action):
                    self.move_left()
            elif action >= 26 and action <= 29:
                for _ in range(action-25):
                    self.move_right()
            self.fall_down()

        elif action >= 30 and action <= 39:
            self.rotate()
            self.rotate()
            self.rotate()
            if action >= 30 and action <= 34:
                for _ in range(35-action):
                    self.move_left()
            elif action >= 36 and action <= 39:
                for _ in range(action-35):
                    self.move_right()
            self.fall_down()

        # If current piece hit ground

        if self.change_piece:
            shape_pos = convert_shape_format(self.curr_piece)
            for pos in shape_pos:
                p = (pos[0], pos[1])
                self.locked_positions[p] = self.curr_piece.color
            self.curr_piece = self.next_piece
            self.next_piece = get_shape()
            self.change_piece = False
            for i in range(4):
                if clear_rows(self.state, self.locked_positions):
                    self.state = create_grid(self.locked_positions)
                    self.score += 10
                    reward += 100
                else:
                    break
        if check_lost(self.locked_positions):
            done = True

        heights = get_height_vs_base(self.state)
        self.observation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.observation[self.curr_piece.shape_index] = 1
        self.observation[self.next_piece.shape_index + 7] = 1
        for i in range(len(heights)):
            self.observation.append(heights[i])
        curr_nb_holes = calculate_hole_nb(self.state)
        reward += (self.nb_holes - curr_nb_holes)*10
        self.nb_holes = curr_nb_holes
        return self.observation, reward, done, info

    def render_bool(self):
        for state in self.state_memo:
            if not self.set_win:
                pygame.font.init()
                self.win = pygame.display.set_mode((s_width, s_height))
                pygame.display.set_caption('XEP HINH CUA THO')
                self.set_win = True
            draw_window(self.win, state)
            draw_next_shape(self.next_piece, self.win)
            draw_score(self.score, self.win)
            pygame.display.update()
            if check_lost(self.locked_positions):
                draw_text_middle("Thua rui Tho lam lai nhe!", 40, (0, 255, 255), self.win)
                pygame.display.update()
            pygame.time.delay(100)

    def reset(self):
        self.locked_positions = {}
        self.change_piece = False
        self.curr_piece = get_shape()
        self.next_piece = get_shape()
        self.state = create_grid(self.locked_positions)
        self.state_memo = [self.state]
        self.score = 0
        self.set_win = False
        self.observation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.observation[self.curr_piece.shape_index] = 1
        self.observation[self.next_piece.shape_index + 7] = 1
        self.nb_holes = 0
        return self.observation

