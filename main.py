import os
import Tetris_env
import gym
from deepQLearing import DeepQNetwork, Agent
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt


#def stack_frames(stacked_frames, frame, buffer_size):
#    if stacked_frames is None:
#        stacked_frames = np.zeros((buffer_size, len(frame)))
#        for idx, _ in enumerate(stacked_frames):
#            stacked_frames[idx,:] = frame
#    else:
#        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
#        stacked_frames[buffer_size-1, :] = frame

#    stacked_frames = stacked_frames.reshape(1, len(frame), buffer_size)
#    return stacked_frames

if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    env = Tetris_env.TetrisEnv()
    load_checkpoint = False
    agent = Agent(gamma=0.3, epsilon=1.0, alpha=0.00025, input_dims=(17),
                  n_actions=40, mem_size=25000, batch_size=64)
    if load_checkpoint:
        agent.load_models()
    filename = 'breakout-alpha0p000025-gamma0p9-only-one-fc-2.png'
    scores = []
    eps_history = []
    numGames = 100000
    stack_size = 4
    score = 0
    performance = []
    # uncomment the line below to record every episode.
    #env = wrappers.Monitor(env, "tmp/breakout-0",
    #                         video_callable=lambda episode_id: True, force=True)
    print("Loading up the agent's memory with random gameplay")
    while agent.mem_cntr < 250:
        done = False
        observation = env.reset()
        observation = np.array(observation).reshape(1, len(observation))
        stacked_frames = None
        #observation = stack_frames(stacked_frames, observation, stack_size)
        while not done:
            action = np.random.randint(0,39)
            observation_, reward, done, info = env.step(action)
            observation_ = np.array(observation_).reshape(1,len(observation_))
            #observation_ = stack_frames(stacked_frames,
            #                            observation_, stack_size)
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
    print("Done with random gameplay. Game on.")
    n_steps = 0
    for i in range(numGames):
        done = False
        #if i % 100 == 0 and i > 0:
        #    x = [j+1 for j in range(i)]

        #    plotLearning(x, scores, eps_history, filename)
        observation = env.reset()
        observation = np.array(observation).reshape(1, len(observation))
        stacked_frames = None
        #observation = stack_frames(stacked_frames, observation, stack_size)
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = np.array(observation_).reshape(1, len(observation_))
            n_steps += 1
            #observation_ = stack_frames(stacked_frames,
            #                            observation_, stack_size)
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            if n_steps % 4 == 0:
                agent.learn()
            if i % 1000 == 0 and i > 1:
                env.render_bool()
        scores.append(score)
        if i % 100 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-100):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.epsilon)
            agent.save_models()
            performance.append(avg_score)
        eps_history.append(agent.epsilon)
        if i % 1000 == 0 and i > 0:
            x = (np.arange(len(performance))+1)*100
            fig = plt.figure()
            plt.plot(x, performance, "b-")
            plt.title("Average score of Tetris AI after {} game".format(len(performance)*100))
            plt.xlabel("Number of game")
            plt.ylabel("Average Score")
            plt.grid()
            plt.savefig(os.getcwd() + "/tmp/plot/avgscore_after_{}_game_ver4.png".format(len(performance)*100))



