# TetrisAI

Reinforcement Learning Tetris game agent using Deep Q network model and Duel Q network model based on pytorch.

Requirement package for running trained weight data: `pytorch`, `gym`, `pygame`.

Requirement package for model training: `matplotlib`, `numpy`

## Idea ##

Tetris game is a game that requires the player to arrange different shapes in the right position to destroy the filled rows.

TetrisAI uses the deep Q network and deep Q network duel respectively to train the agent to play this game.

### Input data:

The agent will receive information about the height of the 10 columns in the play space, information about the shape of the current block, and information about the shape of the next block. There are a total of 7 block shapes. So observation data has 24 dimensions.

### Rewards:

The reward will be calculated by the number of rows that the agent clears in the game screen minus the number of holes (the space between the blocks) created.

## Install dependency ##

### * Install pytorch base on your computer setup
Follow pytorch installation guidance in [Pytorch website](https://pytorch.org/get-started/locally/)

### * Install other package using pip

Gym       : `pip install gym`

Pygame    : `pip install pygame`

Matplotlib: `pip install matplotlib`

Numpy     : `pip install numpy`

## Usage ##

### 1. Run the trained Duel Deep Q learning network weight

Simplely running the python file `test_trained_ddqn.py`

### 2. Run the trained Deep Q learning network weight

Simplely running the python file `test_trained_dqn.py`

### 3. Train the Duel Deep Q learning network model in your computer

Simplely running the python file `ddqn_main.py`

### 4. Train the Deep Q learning network model in your computer

Simplely running the python file `dqn_main.py`


## Result ##

### Deep Q leaning method ###

With Q leaning model, we can get average of `200` points (20 leared rows) per game after 30000 trains game

Result video:


https://user-images.githubusercontent.com/77573775/181213027-740b2a6f-6f46-4568-b2fd-28e94a882f1f.mov



### Duel deep Q learning method ###

With double Q learning model, using one Q for learning and another Q for evaluation, the average score we can achieve is about `400` points (40 cleared rows) per game. Therefore, duel deep Q network perform much better than deep Q network. 

Result video:


https://user-images.githubusercontent.com/77573775/181213802-2225b680-5ecc-459c-8bb0-700b80f5f094.mov


