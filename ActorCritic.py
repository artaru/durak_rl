import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with optional action masking.

    This network features a shared body for feature extraction and two separate
    heads for the policy (actor) and the value function (critic).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initializes the network layers.

        :param obs_dim: The dimension of the observation space.
        :param action_dim: The dimension of the action space.
        :param hidden_dim: The size of the hidden layers.
        """
        super().__init__()

        # Shared network body to extract features from the observation
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

        )

        # The policy head (actor) outputs logits for each action
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )

        # The value head (critic) outputs a single value for the state
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """
        Performs a forward pass through the network.

        :param obs: A tensor of observations.
        :param action_mask: An optional boolean tensor indicating valid actions.
                            Shape: (batch_size, action_dim). True for valid.
        :return: A tuple containing:
                 - dist (torch.distributions.Categorical): The action distribution.
                 - value (torch.Tensor): The estimated state value.
        """
        # Pass observation through the shared network
        shared_features = self.shared_net(obs)

        # --- Actor ---
        action_logits = self.actor_head(shared_features)

        # Apply the action mask, if provided.
        # This is the core of "Maskable" PPO. We set the logits of invalid
        # actions to negative infinity before creating the distribution.
        if action_mask is not None:
            action_logits.masked_fill_(~action_mask.bool(), -float('inf'))

        # Create the action distribution from the (possibly masked) logits
        action_distribution = Categorical(logits=action_logits)

        # --- Critic ---
        state_value = self.critic_head(shared_features)

        return action_distribution, state_value


### Example
if __name__ == "__main__":
    obs_dim = 4
    action_dim = 2
    batch_size = 5


    policy = ActorCritic(obs_dim, action_dim)
    dummy_obs = torch.randn(batch_size, obs_dim)

    # --- Scenario 1: No action mask ---
    print("--- Without Mask ---")
    dist, value = policy(dummy_obs)
    print("Action Logits (unmasked):", dist.logits)
    print("State Values:", value.squeeze())

    action = dist.sample()
    print("Sampled Action:", action)
    log_prob = dist.log_prob(action)
    print("Log Prob of Action:", log_prob)


    # --- Scenario 2: With an action mask ---
    dummy_mask = torch.tensor([
        [True, False],
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ])

    print("\n--- With Mask ---")
    masked_dist, masked_value = policy(dummy_obs, action_mask=dummy_mask)
    print("Action Logits (masked):", masked_dist.logits) # Notice the -inf values
    print("State Values:", masked_value.squeeze())

    masked_action = masked_dist.sample()
    print("Sampled Action (guaranteed to be valid):", masked_action)
    masked_log_prob = masked_dist.log_prob(masked_action)
    print("Log Prob of Action:", masked_log_prob)