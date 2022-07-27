import gym
from tetris_dqn_torch import Agent
import tetris_env
import torch
import os

if __name__ == '__main__':
    env = tetris_env.TetrisEnv()
    load_checkpoint = True

    agent = Agent(gamma=0.9, epsilon=0, lr=5e-4,
                  input_dims=[24], n_actions=40, mem_size=100000, eps_min=0,
                  batch_size=64, eps_dec=0, replace=100)

    if load_checkpoint:
        agent.q_eval.load_state_dict(torch.load(os.path.join(agent.q_eval.checkpoint_dir, "dqn_q_eval_trained")))
        agent.q_next.load_state_dict(torch.load(os.path.join(agent.q_next.checkpoint_dir, "dqn_q_next_trained")))

    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        observation = observation_
        env.render_bool()
    print("Game end with {} point".format(score))
