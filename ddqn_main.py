import gym
import numpy as np
from tetris_ddqn_torch import Agent
import tetris_env
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    env = tetris_env.TetrisEnv()
    num_games = 30000
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-5,
                  input_dims=[24], n_actions=40, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=5e-6, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'Tetris-Dueling-DDQN-256-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)
        if i%100 == 0 and i>1:
            avg_score = np.mean(scores[max(0, i-100):(i+1)])
            print('episode: ', i,'score %.1f ' % score,
                 ' average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, scores, "b-")
    plt.title(filename)
    plt.xlabel("Number of game")
    plt.ylabel("Scores")
    plt.grid()
    plt.savefig(os.getcwd() + "/" + filename)