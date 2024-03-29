import os
import gym
from tetris_dqn_torch import Agent
import numpy as np
import matplotlib.pyplot as plt
import tetris_env

if __name__ == '__main__':
    env = tetris_env.TetrisEnv()
    num_games = 100000
    load_checkpoint = False

    agent = Agent(gamma = 0.99, epsilon = 1.0, lr = 5e-5,
                  input_dims=[24], n_actions = 40, mem_size=10000,
                  eps_min=0.01, batch_size=64, eps_dec=5e-6, replace=100)
    if load_checkpoint:
        agent.load_models()

    filename = 'Tetris-Dueling-DQN-512-Adam-lr0005-replace100.png'
    scores = []
    eps_history = [1.00]
    n_steps = 0
    avg_scores = [0]
    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
            observation = observation_
        scores.append(score)
        if i%100 == 0 and i>1:
            avg_score = np.mean(scores[max(0,i-100):(i+1)])
            avg_scores.append(avg_score)
            print('episode: ', i,'score %.1f ' % score,
                  ' average score %.1f' % avg_score,
                  'epsilon %.2f' % agent.epsilon)
            agent.save_models()
            eps_history.append(agent.epsilon)

    x = [i+1 for i in range(0,num_games,100)]
    fig = plt.figure(figsize=(10,10))
    plt.plot(x, avg_scores,"b-", label = "Average Score")
    plt.plot(x, eps_history*10000, "r-", label = "Epsilon value x 10000")
    plt.xlabel("Number of game")
    plt.grid()
    plt.title("Deep Q Learning Network")
    plt.savefig(os.getcwd() + '/' + filename)



