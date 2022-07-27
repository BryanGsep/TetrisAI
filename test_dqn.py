from tetris_dqn_torch import Agent
import tetris_env


if __name__ == '__main__':
    env = tetris_env.TetrisEnv()
    load_checkpoint = True

    agent = Agent(gamma=0.9, epsilon=0, lr=5e-4,
                  input_dims=[24], n_actions=40, mem_size=100000, eps_min=0,
                  batch_size=64, eps_dec=0, replace=100)

    if load_checkpoint:
        agent.load_models()

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
    