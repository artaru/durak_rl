# Durak AI – Reinforcement Learning Agents
<p align="center">
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/25918a23-ef1b-46d5-b105-33b39920d5eb" />
</p>


This project explores training reinforcement learning agents to play the Russian card game Durak. Using an PPO Actor-Critic framework, agents are trained through self-play and evaluated to measure relative skill progression. The project also includes an interactive mode where humans can play directly against trained AI models.

Durak provides a complex multi-agent environment with imperfect information and evolving strategies, making it an interesting challenge for reinforcement learning implementation.

## Features
* Durak Environment – Custom game environment with rules, actions, and strategy dynamics.
* Actor-Critic Training – Self-play reinforcement learning with policy and value updates.
* Elo Evaluation – Models compared head-to-head with Elo scoring to track progression.
* Human vs AI – Play interactively against trained agents for qualitative evaluation.

## Main Parts
* train.py – Handles model training with Actor-Critic and self-play setup.
* elo_eval.py – Evaluates models against each other using Elo scores.
* play_game.py – Interactive gameplay interface for human vs AI.

