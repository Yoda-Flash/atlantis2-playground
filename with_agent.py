import gymnasium as gym
import ale_py
from agent import Agent

gym.register_envs(ale_py)
env = gym.make("ALE/Atlantis2-v5", render_mode='human')
observation, info = env.reset()

learning_rate = 0.01
n_episodes = 1000
max_iter_per_episode = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = Agent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for _ in range(n_episodes):
    observation = env.reset()
    terminated = False
    total_episode_reward = 0

    for i in range(max_iter_per_episode):
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        # print(observation)
        # print(info)
        # print(reward)

        agent.update(observation, action, reward.__float__(), terminated, next_observation)

        total_episode_reward += reward
        observation = next_observation
        if terminated or truncated:
            observation, info = env.reset()
            break

env.close()