# TetrisAI
Reinforcement Learning Tetris game agent using Q network model based on pytorch

## Install dependency ##

* Install using pip

Tensorflow: [Installation guidance](https://www.tensorflow.org/install/pip)

Gym       : `pip install gym`

Pygame    : `pip install pygame`

Matplotlib: `pip install matplotlib`


## Usage ##

 1. Run the trained duel Deep Q learning network weight

Simplely running the python file `test_trained_ddqn.py`

 2. Train the model in your computer

Simplely running the python file `main.py`

## Result ##

### Deep Q leaning method ###

With Q leaning model, we can get average of `300 points` (3 leared rows) per game after 30000 trains game

Result video:


### Duel deep Q learning method ###

With double Q learning model, using one Q for learning and another Q for evaluation, th average score we can achieve is about `60000` points (60 cleared rows) per game

Result video:

