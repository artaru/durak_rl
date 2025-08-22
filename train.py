import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import os
import copy

from typing import Optional, Dict, Any, List

# Import your existing modules
from ActorCritic import ActorCritic
from MaskableRolloutBuffer import MaskableRolloutBuffer
from durak_env import DurakEnv


class AdaptiveSelfPlayPPOTrainer:
    """
    PPO trainer with adaptive self-play that ensures continued learning by:
    1. Using opponent pool instead of single opponent
    2. Dynamic threshold adjustment
    3. Curriculum learning with weaker opponents
    4. Exploration bonuses and diverse training
    """

    def __init__(
            self,
            env: DurakEnv,
            actor_critic: ActorCritic,
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Optional[float] = None,
            ent_coef: float = 0.02,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: float = 0.01,
            # Adaptive self-play parameters
            initial_threshold: float = 0.55,
            eval_episodes: int = 50,
            eval_interval: int = 15,
            min_updates_before_eval: int = 5,
            # Opponent pool parameters
            pool_size: int = 5,
            random_opponent_prob: float = 0,
            curriculum_prob: float = 0.4,

            diversity_reward: float = 0.005,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the adaptive self-play PPO trainer with multiple learning enhancements.
        """
        self.env = env
        self.actor_critic = actor_critic.to(device)
        self.device = device

        # PPO Hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf if clip_range_vf is not None else clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Adaptive self-play parameters
        self.current_threshold = initial_threshold
        self.eval_episodes = eval_episodes
        self.eval_interval = eval_interval
        self.min_updates_before_eval = min_updates_before_eval

        # Opponent pool parameters
        self.pool_size = pool_size
        self.random_opponent_prob = random_opponent_prob
        self.curriculum_prob = curriculum_prob

        # Learning enhancement
        self.diversity_reward = diversity_reward

        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Initialize opponent pool
        self.opponent_pool: List[nn.Module] = []
        self.opponent_strengths: List[float] = []  # Track relative strength
        self.current_opponent_idx = 0

        # Add initial opponent (copy of main network)
        initial_opponent = copy.deepcopy(self.actor_critic)
        initial_opponent.eval()
        self.opponent_pool.append(initial_opponent)
        self.opponent_strengths.append(0.5)  # Neutral strength

        # Set up opponent policy wrapper
        self.setup_opponent_policy()

        # Initialize rollout buffer
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.rollout_buffer = MaskableRolloutBuffer(
            buffer_size=n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gae_lambda=gae_lambda,
            gamma=gamma
        )

        # Training and self-play metrics
        self.episode_rewards = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.win_rates = []
        self.thresholds = []
        self.opponent_updates = []
        self.evaluation_history = []
        self.opponent_selection_history = []

        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': []
        }

        # Self-play tracking
        self.updates_since_opponent_change = 0
        self.updates_since_last_reevaluation = 0
        self.total_opponent_updates = 0
        self.last_win_rates = deque(maxlen=5)  # Track recent win rates

        print(f"Adaptive Self-play PPO initialized:")
        print(f"  Initial threshold: {initial_threshold:.1%}")
        print(f"  Opponent pool size: {pool_size}")
        print(f"  Random opponent prob: {random_opponent_prob:.1%}")
        print(f"  Curriculum prob: {curriculum_prob:.1%}")

    def setup_opponent_policy(self):
        """Create a policy wrapper for the current opponent."""

        class OpponentPolicy:
            def __init__(self, trainer):
                self.trainer = trainer

            def predict(self, obs, action_masks=None, deterministic=True):
                # Select opponent based on probabilities
                opponent = self.trainer.select_opponent_for_training()

                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.trainer.device)
                    mask_tensor = torch.BoolTensor(action_masks).unsqueeze(0).to(
                        self.trainer.device) if action_masks is not None else None

                    action_dist, _ = opponent(obs_tensor, mask_tensor)

                    # Add some randomness even in deterministic mode for diversity
                    if deterministic and np.random.random() < 0.1:  # 10% random actions
                        valid_actions = np.where(action_masks)[0] if action_masks is not None else list(
                            range(action_dist.logits.shape[-1]))
                        action = np.random.choice(valid_actions)
                        action = torch.tensor([action])
                    else:
                        action = action_dist.sample()

                    return action.cpu().numpy()[0], None

        opponent_policy = OpponentPolicy(self)
        self.env.set_opponent(opponent_policy)

    def select_opponent_for_training(self) -> nn.Module:
        """Select an opponent for training based on curriculum and diversity."""
        rand_val = np.random.random()

        if rand_val < self.random_opponent_prob:
            # Random opponent from pool
            idx = np.random.randint(len(self.opponent_pool))
            self.opponent_selection_history.append(('random', idx))
            return self.opponent_pool[idx]
        elif rand_val < self.random_opponent_prob + self.curriculum_prob:
            if len(self.opponent_pool) > 1:
                # Select from weaker half of opponents
                sorted_indices = np.argsort(self.opponent_strengths)
                weak_half = sorted_indices[:len(sorted_indices) // 2 + 1]
                idx = np.random.choice(weak_half)
                self.opponent_selection_history.append(('curriculum', idx))
                return self.opponent_pool[idx]
            else:
                idx = 0
                self.opponent_selection_history.append(('curriculum', idx))
                return self.opponent_pool[idx]
        else:
            # Use current strongest opponent
            strongest_idx = np.argmax(self.opponent_strengths)
            self.opponent_selection_history.append(('strongest', strongest_idx))
            return self.opponent_pool[strongest_idx]

    def evaluate_against_opponent(self, specific_opponent_idx: Optional[int] = None) -> Dict[str, float]:
        """Evaluate current model against a specific opponent or the strongest one."""
        if specific_opponent_idx is not None:
            opponent = self.opponent_pool[specific_opponent_idx]
            eval_name = f"opponent_{specific_opponent_idx}"
        else:
            # Evaluate against strongest opponent
            strongest_idx = np.argmax(self.opponent_strengths)
            opponent = self.opponent_pool[strongest_idx]
            eval_name = f"strongest_opponent_{strongest_idx}"

        print(f"Evaluating against {eval_name} over {self.eval_episodes} episodes...")

        wins = 0
        losses = 0
        episode_rewards = []
        episode_lengths = []

        self.actor_critic.eval()

        # Temporarily set specific opponent
        original_env_opponent = self.env.opponent_policy

        class TempOpponentPolicy:
            def __init__(self, network, device):
                self.network = network
                self.device = device

            def predict(self, obs, action_masks=None, deterministic=True):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    mask_tensor = torch.BoolTensor(action_masks).unsqueeze(0).to(
                        self.device) if action_masks is not None else None

                    action_dist, _ = self.network(obs_tensor, mask_tensor)
                    action = action_dist.sample()

                    return action.cpu().numpy()[0], None

        temp_opponent = TempOpponentPolicy(opponent, self.device)
        self.env.set_opponent(temp_opponent)

        for episode in range(self.eval_episodes):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)

            episode_reward = 0
            episode_length = 0

            while True:
                action_mask = self.env.action_masks()
                action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_dist, _ = self.actor_critic(obs.unsqueeze(0), action_mask_tensor)
                    action = action_dist.sample()
                    action_np = action.cpu().numpy()[0]

                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)

                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    break
                else:
                    obs = torch.FloatTensor(next_obs).to(self.device)

        # Restore original opponent policy
        self.env.set_opponent(original_env_opponent)

        win_rate = wins / self.eval_episodes

        loss_rate = losses / self.eval_episodes

        eval_results = {
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'mean_reward': np.mean(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'wins': wins,
            'losses': losses,
            'opponent_idx': specific_opponent_idx or np.argmax(self.opponent_strengths)
        }

        self.evaluation_history.append(eval_results)
        self.win_rates.append(win_rate)
        self.last_win_rates.append(win_rate)

        print(f"Evaluation vs {eval_name}: Win: {win_rate:.1%}, Loss: {loss_rate:.1%}")

        return eval_results

    # Add this method inside the AdaptiveSelfPlayPPOTrainer class

    def reevaluate_opponent_strengths(self):
        """
        Re-evaluates the entire opponent pool against the current agent to update strength scores.
        """
        print("\n" + "-" * 40)
        print(" Re-evaluating entire opponent pool strengths...")

        if not self.opponent_pool:
            return

        updated_strengths = []
        # We evaluate against all opponents, including the one at index 0
        for i in range(len(self.opponent_pool)):
            # We don't need the full dictionary, just the win rate
            eval_results = self.evaluate_against_opponent(specific_opponent_idx=i)
            win_rate = eval_results['win_rate']

            # You could use a moving average to smooth the strength updates
            # For simplicity here, we'll just replace it.
            old_strength = self.opponent_strengths[i]
            new_strength = 0.5 * old_strength + 0.5 * (1 - win_rate) # Opponent strength is inverse of our win_rate



            updated_strengths.append(new_strength)

        self.opponent_strengths = updated_strengths
        self.updates_since_opponent_change = 0
        print(f" Opponent strengths updated: {[f'{s:.2f}' for s in self.opponent_strengths]}")
        print("-" * 40 + "\n")

    def update_opponent_pool(self):
        """Add current model to opponent pool and manage pool size."""
        print(" Adding current model to opponent pool...")

        # Create new opponent
        new_opponent = copy.deepcopy(self.actor_critic)
        new_opponent.eval()

        # Add to pool
        self.opponent_pool.append(new_opponent)

        # Estimate strength based on current win rate
        current_strength = np.mean(self.last_win_rates) if self.last_win_rates else 0.5
        self.opponent_strengths.append(current_strength)

        # Manage pool size - remove weakest if too large
        if len(self.opponent_pool) > self.pool_size:
            weakest_idx = np.argmin(self.opponent_strengths)
            self.opponent_pool.pop(weakest_idx)
            self.opponent_strengths.pop(weakest_idx)

        self.total_opponent_updates += 1
        self.updates_since_opponent_change = 0
        self.opponent_updates.append(len(self.training_metrics['policy_loss']))

        print(f" Opponent pool updated! Pool size: {len(self.opponent_pool)}")
        print(f" Opponent strengths: {[f'{s:.2f}' for s in self.opponent_strengths]}")
        print(f" Total updates: {self.total_opponent_updates}")

    def collect_rollouts(self) -> Dict[str, float]:
        """Collect experiences"""
        self.actor_critic.eval()

        # Reset buffer
        self.rollout_buffer.pos = 0
        self.rollout_buffer.full = False

        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        episode_reward = 0
        episode_length = 0
        episodes_completed = 0

        # Track action diversity for bonus
        action_counts = np.zeros(self.env.action_space.n)

        for step in range(self.n_steps):
            action_mask = self.env.action_masks()
            action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_dist, value = self.actor_critic(obs.unsqueeze(0), action_mask_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0]

            # Track action for diversity
            action_counts[action_np] += 1

            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            # Enhanced reward shaping
            shaped_reward = reward

            # Diversity reward based on entropy of action distribution
            if step > 50:  # After some steps
                action_entropy = -np.sum((action_counts + 1e-8) / np.sum(action_counts + 1e-8) *
                                         np.log(action_counts + 1e-8))
                diversity_bonus = self.diversity_reward * action_entropy
                shaped_reward += diversity_bonus

            self.rollout_buffer.add(
                obs=obs.cpu().numpy(),
                action=action_np,
                reward=shaped_reward,  # Use shaped reward
                done=done,
                log_prob=log_prob_np,
                value=value_np,
                action_mask=action_mask
            )

            episode_reward += reward  # Track original reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episodes_completed += 1

                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)
                episode_reward = 0
                episode_length = 0
            else:
                obs = torch.FloatTensor(next_obs).to(self.device)

        # Get final value
        action_mask = self.env.action_masks()
        action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, last_value = self.actor_critic(obs.unsqueeze(0), action_mask_tensor)
            last_value = last_value.cpu()

        self.rollout_buffer.compute_returns_and_advantage(
            last_value=last_value,
            last_done=done
        )

        return {
            'episodes_completed': episodes_completed,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }

    def train(self) -> Dict[str, float]:
        """Enhanced training with adaptive learning rate and regularization."""
        self.actor_critic.train()

        # Training metrics for this update
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        clip_fractions = []


        for epoch in range(self.n_epochs):
            for batch_data in self.rollout_buffer.get(self.batch_size):
                obs_b, actions_b, old_log_probs_b, advantages_b, returns_b, action_masks_b = batch_data

                # Move to device
                obs_b = obs_b.to(self.device)
                actions_b = actions_b.to(self.device).long()
                old_log_probs_b = old_log_probs_b.to(self.device)
                advantages_b = advantages_b.to(self.device)
                returns_b = returns_b.to(self.device)
                action_masks_b = action_masks_b.to(self.device)

                # Normalize advantages for this batch
                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

                # Get current policy outputs
                action_dist, values = self.actor_critic(obs_b, action_masks_b)

                # Calculate policy loss
                log_probs = action_dist.log_prob(actions_b)
                ratio = torch.exp(log_probs - old_log_probs_b)

                # PPO clipped objective
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                values = values.squeeze()
                value_loss = nn.MSELoss()(values, returns_b)

                # Enhanced entropy loss (important for continued exploration)
                entropy_loss = -action_dist.entropy().mean()

                # Total loss
                total_loss = (
                        policy_loss +
                        self.vf_coef * value_loss +
                        self.ent_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Calculate metrics
                with torch.no_grad():
                    kl_div = (old_log_probs_b - log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_divergences.append(kl_div.item())
                clip_fractions.append(clip_fraction.item())

                # Early stopping
                if kl_div > self.target_kl * 1.5:
                    break

            if np.mean(kl_divergences[-10:]) > self.target_kl * 1.5:
                break

        # Calculate explained variance
        y_pred = self.rollout_buffer.values
        y_true = self.rollout_buffer.returns
        var_y = torch.var(y_true)
        explained_var = 1 - torch.var(y_true - y_pred) / var_y if var_y > 0 else 0

        training_stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(policy_losses) + self.vf_coef * np.mean(value_losses) + self.ent_coef * np.mean(
                entropy_losses),
            'kl_divergence': np.mean(kl_divergences),
            'clip_fraction': np.mean(clip_fractions),
            'explained_variance': explained_var.item() if torch.is_tensor(explained_var) else explained_var
        }

        # Update training metrics history
        for key, value in training_stats.items():
            self.training_metrics[key].append(value)

        return training_stats

    def learn(self, total_timesteps: int, log_interval: int = 10, save_interval: int = 100, save_path: str = "models/"):
        """Main training loop with adaptive self-play."""
        os.makedirs(save_path, exist_ok=True)

        total_updates = total_timesteps // self.n_steps

        print(f"Starting Adaptive Self-Play PPO training for {total_timesteps} timesteps ({total_updates} updates)")
        print("-" * 120)

        for update in range(1, total_updates + 1):

            rollout_stats = self.collect_rollouts()
            training_stats = self.train()

            self.updates_since_opponent_change += 1
            self.updates_since_last_reevaluation += 1

            should_evaluate = (
                    update >= self.min_updates_before_eval and
                    update % self.eval_interval == 0
            )

            eval_results = None
            opponent_updated = False

            if should_evaluate:
                eval_results = self.evaluate_against_opponent()

                if eval_results['win_rate'] >= self.current_threshold:
                    self.update_opponent_pool()
                    opponent_updated = True
                    self.reevaluate_opponent_strengths()


            # Logging
            if update % log_interval == 0 or update == 1 or opponent_updated:

                print(
                    f"Update {update}/{total_updates} | Pool size: {len(self.opponent_pool)}")
                print(f"  Episodes: {rollout_stats['episodes_completed']}")
                print(f"  Mean reward: {rollout_stats['mean_episode_reward']:.3f}")
                print(f"  Policy loss: {training_stats['policy_loss']:.6f}")
                print(f"  Entropy: {training_stats['entropy_loss']:.6f}")
                print(f"  Updates since pool change: {self.updates_since_opponent_change}")

                if eval_results:
                    print(f" Win rate: {eval_results['win_rate']:.1%}")
                    if opponent_updated:
                        print(f" OPPONENT POOL UPDATED!")

                print("-" * 120)

            # Save model
            if update % save_interval == 0:
                model_path = os.path.join(save_path, f"adaptive_selfplay_ppo_update_{update}.pt")

                torch.save({
                    'actor_critic_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'opponent_pool_state_dicts': [opp.state_dict() for opp in self.opponent_pool],
                    'opponent_strengths': self.opponent_strengths,
                    'current_threshold': self.current_threshold,
                    'update': update,
                    'training_metrics': self.training_metrics,
                    'evaluation_history': self.evaluation_history,
                    'opponent_updates': self.opponent_updates,
                    'win_rates': self.win_rates,
                    'thresholds': self.thresholds,
                    'total_opponent_updates': self.total_opponent_updates
                }, model_path)

                print(f"Model and opponent pool saved: {model_path}")

        print("Adaptive self-play training completed!")
        print(f"Final opponent pool size: {len(self.opponent_pool)}")
        print(f"Total opponent updates: {self.total_opponent_updates}")
        print(f"Final threshold: {self.current_threshold:.1%}")

        return self.training_metrics

    def plot_training_metrics(self, save_path: str = "adaptive_selfplay_plots.png"):
        """Enhanced plotting with self-play metrics."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Adaptive Self-Play PPO Training Metrics', fontsize=16, y=1.02)


        axes = axes.flatten()
        training_metrics = [
            ('policy_loss', 'Policy Loss', 0),
            ('value_loss', 'Value Loss', 1),
            ('entropy_loss', 'Entropy Loss (Higher = More Exploration)', 2),
            ('kl_divergence', 'KL Divergence', 3)
        ]

        for metric, title, ax_idx in training_metrics:
            if metric in self.training_metrics and self.training_metrics[metric]:
                axes[ax_idx].plot(self.training_metrics[metric])
                axes[ax_idx].set_title(title)
                axes[ax_idx].set_xlabel('Update')
                axes[ax_idx].grid(True)

        if hasattr(self, 'win_rates') and self.win_rates:
            ax = axes[4]
            ax.plot(range(len(self.win_rates)), self.win_rates, 'g-', linewidth=2, label='Win Rate')
            ax.set_title('Win Rate')
            ax.set_xlabel('Evaluation')
            ax.set_ylabel('Rate')
            ax.legend()
            ax.grid(True)


        ax = axes[5]
        if hasattr(self, 'pool_size_history') and self.pool_size_history:
            ax.plot(self.pool_size_history)
            ax.set_title('Opponent Pool Size')
            ax.set_xlabel('Update')
            ax.set_ylabel('Pool Size')
            ax.grid(True)
        elif hasattr(self, 'opponent_strengths') and self.opponent_strengths:
            ax.bar(range(len(self.opponent_strengths)), self.opponent_strengths,
                   alpha=0.7, color='skyblue')
            ax.set_title('Current Opponent Pool Strengths')
            ax.set_xlabel('Opponent Index')
            ax.set_ylabel('Estimated Strength')
            ax.grid(True)
        else:
            # Hide the unused subplot
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the adaptive self-play training."""

    env = DurakEnv()


    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor_critic = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)

    # Initialize adaptive self-play trainer
    trainer = AdaptiveSelfPlayPPOTrainer(
        env=env,
        actor_critic=actor_critic,
        learning_rate=5e-4,
        n_steps=1000,
        batch_size=128,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.97,
        clip_range=0.25,
        ent_coef=0.01,
        vf_coef=0.7,
        max_grad_norm=0.5,
        target_kl=0.015,
        initial_threshold=0.55,
        eval_episodes=150,
        eval_interval=15,
        min_updates_before_eval=5,
        pool_size=10,
        random_opponent_prob=0.5,
        curriculum_prob=0.4,
        diversity_reward=0.001
    )

    # Start adaptive self-play training
    total_timesteps = 1_000_000  #
    training_metrics = trainer.learn(
        total_timesteps=total_timesteps,
        log_interval=20,
        save_interval=50
    )

    # Plot comprehensive results
    trainer.plot_training_metrics()

    # Print final training summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total opponent pool updates: {trainer.total_opponent_updates}")
    print(f"Final opponent pool size: {len(trainer.opponent_pool)}")
    print(f"Final threshold: {trainer.current_threshold:.1%}")
    print(f"Final win rate: {trainer.win_rates[-1]:.1%}" if trainer.win_rates else "No evaluations")
    print(f"Opponent strengths: {[f'{s:.2f}' for s in trainer.opponent_strengths]}")

    if len(trainer.win_rates) > 5:
        recent_improvement = max(trainer.win_rates[-5:]) - min(trainer.win_rates[-5:])
        print(f"Recent improvement range: {recent_improvement:.1%}")
    return trainer

if __name__ == "__main__":
    trainer = main()