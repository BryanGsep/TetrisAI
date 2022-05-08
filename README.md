# TetrisAI
Using Reinforcement learning to learn how to play Tetris game with Q learning model

The code requires '''tensorflow''', '''gym''', '''pygame''', and '''matplotlib''' package for running.

Execute main.py file to run the whole program.

## Idea
* Environment:
The environment include 17 observation variables with 7 variables for current piece shape and 10 variables for height of 10 collums.

* Action:
For each piece, I decide to have 40 types of motion (4 rotations at the beginning of the game * 10 locations)

* Tensorflow model:
The neuron network model is simple with one input layer (17 nodes), one hidden layer (512 nodes) and output layer (40 nodes)

* Learning model:
Q learning model with learing fuction is Q(s,a) = Q(s,a) + learing_rate*( gamma*max_a(Q(s+1,:))*terminal - Q(s,a))

* Loss function:
Reduce mean of (Q_values - q_targets)

* Train function:
Adam optimizer

## Result
With this reinforcement model, we can get average of 300 points (3 leared rows) per game after 30000 trains game
