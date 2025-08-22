import torch
import numpy as np


class MaskableRolloutBuffer:
    """
    A rollout buffer that stores experiences for PPO and supports action masking.
    It also calculates the Generalized Advantage Estimation (GAE).
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, gae_lambda: float = 0.95, gamma: float = 0.99):
        """
        Initializes the buffer.

        :param buffer_size: The number of steps to store per rollout (n_steps).
        :param obs_dim: The dimension of the observation space.
        :param action_dim: The dimension of the action space.
        :param gae_lambda: The lambda factor for GAE.
        :param gamma: The discount factor.
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # Initialize storage tensors
        self.observations = torch.zeros((buffer_size, obs_dim))
        self.actions = torch.zeros((buffer_size,))
        self.rewards = torch.zeros((buffer_size,))
        self.dones = torch.zeros((buffer_size,))
        self.log_probs = torch.zeros((buffer_size,))
        self.values = torch.zeros((buffer_size,))
        self.action_masks = torch.zeros((buffer_size, action_dim), dtype=torch.bool)

        # These will be computed after the rollout is complete
        self.advantages = torch.zeros((buffer_size,))
        self.returns = torch.zeros((buffer_size,))

        self.pos = 0  # Current position in the buffer
        self.full = False

    def add(self, obs, action, reward, done, log_prob, value, action_mask):
        """Adds a single transition to the buffer."""
        if self.pos >= self.buffer_size:
            self.pos = 0  # Reset position if buffer is full
            self.full = True

        self.observations[self.pos] = torch.as_tensor(obs)
        self.actions[self.pos] = torch.as_tensor(action)
        self.rewards[self.pos] = torch.as_tensor(reward)
        self.dones[self.pos] = torch.as_tensor(done)
        self.log_probs[self.pos] = torch.as_tensor(log_prob)
        # Squeeze value in case it's a tensor of shape [1, 1]
        self.values[self.pos] = torch.as_tensor(value).squeeze()
        self.action_masks[self.pos] = torch.as_tensor(action_mask)

        self.pos += 1

    def compute_returns_and_advantage(self, last_value: torch.Tensor, last_done: bool):
        """
        Computes the returns and advantages for the entire buffer.
        This should be called after the buffer is full.

        :param last_value: The value estimate for the state after the last one in the buffer.
        :param last_done: Whether the episode was done after the last state.
        """
        last_value = last_value.clone().detach().squeeze()
        advantage = 0

        # Iterate backwards through the buffer to calculate GAE
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # Calculate the TD error (delta)
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]

            # Calculate the advantage using GAE formula
            advantage = delta + self.gamma * self.gae_lambda * advantage * next_non_terminal
            self.advantages[t] = advantage

        # The returns are simply the advantages plus the value function estimates
        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        """
        Returns a generator that yields random mini-batches from the buffer.

        :param batch_size: The size of each mini-batch.
        """
        # Ensure the buffer has been completely filled
        num_samples = self.buffer_size
        indices = np.random.permutation(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            minibatch_indices = indices[start_idx:end_idx]

            yield (
                self.observations[minibatch_indices],
                self.actions[minibatch_indices],
                self.log_probs[minibatch_indices],
                self.advantages[minibatch_indices],
                self.returns[minibatch_indices],
                self.action_masks[minibatch_indices],
            )


### Usage Example
if __name__ == "__main__":
    # --- Parameters ---
    buffer_size = 20
    obs_dim = 4
    action_dim = 2
    batch_size = 5

    # --- Instantiate ---
    buffer = MaskableRolloutBuffer(buffer_size, obs_dim, action_dim)

    # --- Simulate filling the buffer with dummy data ---
    for i in range(buffer_size):
        buffer.add(
            obs=np.random.rand(obs_dim),
            action=np.random.randint(0, action_dim),
            reward=np.random.rand(),
            done=False,  # Assume not done for simplicity
            log_prob=-0.6,
            value=np.random.rand(),
            action_mask=np.random.choice([True, False], size=action_dim)
        )

    print(f"Buffer position: {buffer.pos}. Is buffer full? {buffer.pos == buffer_size}")

    # --- Compute returns and advantages ---

    last_value_estimate = torch.tensor(0.5)
    buffer.compute_returns_and_advantage(last_value_estimate, last_done=False)

    print("\nAdvantages sample:", buffer.advantages[:5].numpy().round(2))
    print("Returns sample:", buffer.returns[:5].numpy().round(2))

    print(f"\nGetting mini-batches of size {batch_size}...")
    batch_generator = buffer.get(batch_size)
    first_batch = next(batch_generator)

    obs_b, act_b, logp_b, adv_b, ret_b, mask_b = first_batch

    print("\nShape of tensors in the first mini-batch:")
    print(f"Observations: {obs_b.shape}")
    print(f"Actions: {act_b.shape}")
    print(f"Log Probs: {logp_b.shape}")
    print(f"Advantages: {adv_b.shape}")
    print(f"Returns: {ret_b.shape}")
    print(f"Action Masks: {mask_b.shape}")