# %% [markdown]
# # PPO - Take 2

# %%
from collections import deque, namedtuple
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F
import tqdm
import pickle
import faulthandler

# %% [markdown]
# ## Utility functions

# %%
def moving_average(data, *, window_size = 50):
    """Smooths 1-D data array using a moving average.

    Args:
        data: 1-D numpy.array
        window_size: Size of the smoothing window

    Returns:
        smooth_data: A 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

# %%
def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title, smoothing = True, xlabel="Time Steps"):
    """
    Args:
        arr_list (list): List of results arrays to plot
        legend_list (list): List of legends corresponding to each result array
        color_list (list): List of color corresponding to each result array
        ylabel (string): Label of the vertical axis

        Make sure the elements in the arr_list, legend_list, and color_list
        are associated with each other correctly (in the same order).
        Do not forget to change the ylabel for different plots.
    """
    # Set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    # PLEASE NOTE: Change the vertical labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Plot results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # Compute the standard error (of raw data, not smoothed)
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # Plot the mean
        averages = moving_average(arr.mean(axis=0)) if smoothing else arr.mean(axis=0)
        h, = ax.plot(range(arr.shape[1]), averages, color=color, label=legend)
        # Plot the confidence band
        arr_err *= 1.96
        print(f"{(averages - arr_err).shape=}")
        ax.fill_between(range(arr.shape[1]), averages - arr_err, averages + arr_err, alpha=0.3,
                        color=color)
        # Save the plot handle
        h_list.append(h)

    # Plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    plt.show()

# %%
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

# %%
def compute_grad_magnitudes(model):
    """Compute the total gradient magnitude for a model's gradients."""
    total_grad = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad += torch.norm(param.grad).item()
    return total_grad

# %%
def update_parent_info(parent_info, child_info, extend=False):
    for key, val in child_info.items():
        parent_info.setdefault(key, [])
        if extend and isinstance(val, list):
            parent_info[key].extend(val)
        else:
            parent_info[key].append(val)

# %%
def save_object(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)

# %% [markdown]
# ## PPO Implementation

# %% [markdown]
# ### Actor and Critic Networks

# %%
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_hidden=1, softmax=False):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim), # Input layer
            nn.ReLU()
        ]

        # Additional hidden layers
        if num_hidden > 1:
            for _ in range(num_hidden - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim)) # Output layer

        if softmax:
            layers.append(nn.Softmax(dim=-1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

# %%
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_hidden=1):
        super().__init__()

        self.actor = MLP(state_dim, action_dim, hidden_dim, num_hidden, softmax=True)
        self.critic = MLP(state_dim, 1, hidden_dim, num_hidden)

    def forward(self, x):
        state_value = self.critic(x)
        action_probs = self.actor(x)

        return state_value, action_probs

# %% [markdown]
# ### PPO Agent

# %%
class PPOAgent(object):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_hidden=1):
        # Create the actor and critic networks
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim, num_hidden=1)
        self.actor_critic.apply(init_weights)

    def get_action(self, state):
        # Sample an action from the actor network, return the action and its log probability,
        # and return the state value according to the critic network
        state_tensor = torch.tensor(state).float().view(1, -1)
        state_value, action_probs = self.actor_critic(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, state_value

    def evaluate(self, states, actions):
        """
        Returns state value estimate and entropy of action distribution across a batch of states
        and log probability of taking given actions at those states
        """

        state_values, action_probs = self.actor_critic(states)
        m = Categorical(action_probs)

        return m.entropy(), m.log_prob(actions), state_values

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

# %% [markdown]
# ### Replay Buffer

# %%
Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones', 'log_probs', 'state_values', 'returns')
)


class ReplayMemory:
    def __init__(self, max_size, state_dim):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer
            - state_dim: Size of the state-space features for the environment
        """
        self.max_size = max_size
        self.state_dim = state_dim

        # Preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_dim))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_dim))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)
        # self.boostrap_returns = torch.empty((max_size, 1))
        self.log_probs = torch.empty((max_size, 1))
        self.state_values = torch.empty((max_size, 1))

        # to store rewards-to-go, computed later
        self.returns = torch.empty((max_size, 1))
        self.returns_computed = False

        # Pointer to the current location in the circular buffer
        self.idx = 0
        # Indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done, log_prob, state_value):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        :param log_prob log of probability for selected action
        :param state_value s
        """

        if self.size == self.max_size:
            raise Exception('Buffer full!')

        # YOUR CODE HERE: Store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`

        self.states[self.idx] = torch.tensor(state)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = torch.tensor(next_state)
        self.dones[self.idx] = done

        # self.boostrap_returns[self.idx] = bootstrap_return
        self.log_probs[self.idx] = log_prob
        self.state_values[self.idx] = state_value

        # Increment the index
        self.idx += 1
        # Update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def compute_returns(self, gamma, normalize=True):
        G = 0
        ep_returns = deque()

        for i in range(self.size - 1, -1, -1):
            reward = self.rewards[i]
            done = self.dones[i]

            if done:
                # end of episode, reset discounted return calculation
                ep_returns.appendleft(G)
                G = 0

            # always detach returns, never take gradient w.r.t return
            G = gamma * G + reward.detach().item()
            self.returns[i] = G

        if normalize:
            returns = self.returns[:self.size]
            self.returns[:self.size] = (returns - returns.mean()) / (returns.std() + 1e-10)

        self.returns_computed = True

        return G, ep_returns

    def get_all(self):
        if not self.returns_computed:
            raise Exception('Returns not yet computed! Please call buffer.compute_returns() before sampling.')

        return Batch(
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.next_states[:self.size],
            self.dones[:self.size],
            self.log_probs[:self.size],
            self.state_values[:self.size],
            self.returns[:self.size]
        )


    def batches(self, batch_size):
        """Iterate through random batches of sampled transitions.

        :param batch_size: Number of transitions to sample in each batch
        :rtype: Batch
        """

        if not self.returns_computed:
            raise Exception('Returns not yet computed! Please call buffer.compute_returns() before sampling.')

        shuf_indices = torch.randperm(self.size, dtype=torch.long)

        for start_idx in range(0, self.size, batch_size):
            end_idx = start_idx + batch_size

            if end_idx > self.size:
                end_idx = self.size

            sample_indices = shuf_indices[start_idx:end_idx]

            batch = Batch(
                self.states[sample_indices],
                self.actions[sample_indices],
                self.rewards[sample_indices],
                self.next_states[sample_indices],
                self.dones[sample_indices],
                self.log_probs[sample_indices],
                self.state_values[sample_indices],
                self.returns[sample_indices]
            )

            yield batch

    def clear(self):
        self.idx = 0
        self.size = 0
        self.returns_computed = False

# %% [markdown]
# ### Training Loop

# %%
class PPOAgentTrainer(object):
    def __init__(self, agent: PPOAgent, env: gym.Env, params):
        # Agent object
        self.agent = agent

        # Environment object
        self.env = env

        # Training parameters
        self.params = params

        # Lists to store the log probabilities, state values, and rewards for one episode
        self.buffer = ReplayMemory(max_size=params['buffer_size'], state_dim=params['state_dim'])

        # Hyperparameters

        self.gamma = params['gamma'] # Gamma - discount factor
        self.epsilon = params['clip_epsilon'] # epsilon for PPO-Clip
        self.k_epochs = params['epochs_per_update'] # how many optimizer steps every PPO update
        self.entropy_coef = params['entropy_coef'] # entropy bonus coefficient

        self.optimizer_actor = torch.optim.Adam(agent.actor_critic.actor.parameters(), lr=params['actor_learning_rate'])
        self.optimizer_critic = Adam(agent.actor_critic.critic.parameters(), lr=params['critic_learning_rate'])

    def ppo_one_epoch(self, states, actions, returns, state_values, old_log_probs, advantage):
        entropy_bonus, current_log_probs, latest_state_values = self.agent.evaluate(states, actions)

        # Probability ratio of selecting action at s, use log_prob for pi_theta_old
        # We take the actor gradient w.r.t current_log_probs
        ratio_t = torch.exp(current_log_probs - old_log_probs) # equivalent to e^current_log_probs / e^old_log_probs

        # Unclipped surrogate advantage
        unclipped_adv = ratio_t * advantage

        # Clipped surrogate advantage
        clipped_adv = torch.clamp(ratio_t, min=(1 - self.epsilon), max=(1 + self.epsilon))

        # Choose the minimum of the two (in the negative direction, if we choose a bad action, it should be bad)
        # unclipped chosen when ratio > 1+epsilon and advantage is negative
        # or when ratio < 1 - epsilon and advantage is positive
        # negative sign because torch Adam minimizes
        policy_loss = -torch.min(unclipped_adv, clipped_adv).mean()

        assert not torch.isnan(old_log_probs).any(), "Found NaN in log_probs"
        assert not torch.isnan(ratio_t).any(), "Found NaN in ratio_t"
        assert not torch.isnan(entropy_bonus).any(), "Found NaN in entropy_bonus"

        # Critic / value function estimate loss based on monte carlo return
        # value loss gradient is w.r.t. latest_state_values
        value_loss = F.mse_loss(latest_state_values, returns).mean()

        # Incorporate entropy bonus
        policy_loss -= (self.entropy_coef * entropy_bonus.mean())


        # Gradient descent
        # ================
        # Optimize critic w.r.t value loss
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        # critic_grad_mag = compute_grad_magnitudes(self.agent.policy_net.groups.critic)
        # if LOG_LEVEL == 'debug' or LOG_LEVEL == 'grads':

        #     print(f'Critic gradient magnitudes: {critic_grad_mag}')
        self.optimizer_critic.step()

        # Optimize actor w.r.t. policy loss
        self.optimizer_actor.zero_grad()
        policy_loss.backward()

        # print gradients
        # actor_grad_mag = compute_grad_magnitudes(self.agent.policy_net.groups.actor)
        # if LOG_LEVEL == 'debug' or LOG_LEVEL == 'grads':
        #     print(f'Actor gradient magnitudes: {actor_grad_mag}')
        self.optimizer_actor.step()

        return policy_loss, value_loss, {}

    def ppo_update(self):
        # def step_ppo_update(self, dataset: Batch):
        # Detach gradients for all collected data - don't compute gradients w.r.t them
        G, _ = self.buffer.compute_returns(self.gamma) # Compute returns for the collected episode
        data = self.buffer.get_all()

        states = data.states.detach()
        actions = data.actions.detach()
        returns = data.returns.detach()
        state_values = data.state_values.detach()
        old_log_probs = data.log_probs.detach()

        # Compute advantages based on MC returns
        advantage = (returns - state_values)

        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10) # add constant to prevent div by 0

        total_policy_loss = 0
        total_value_loss = 0

        info = {
            'episode_return': G
        }

        for epoch in range(self.k_epochs):
            epoch_policy_loss, epoch_value_loss, epoch_info = self.ppo_one_epoch(
                states, actions, returns, state_values, old_log_probs, advantage
                )

            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            update_parent_info(info, epoch_info)

        # Detach gradients, these are only used for tracking statistics
        mean_policy_loss = total_policy_loss.detach() / self.k_epochs
        mean_value_loss = total_value_loss.detach() / self.k_epochs

        info['actor_loss'] = mean_policy_loss
        info['critic_loss'] = mean_value_loss
        info['combined_loss'] = mean_policy_loss + mean_value_loss

        self.buffer.clear() # collected data no longer valid

        return info

    def rollout(self):
        # Roll out one episode
        state, info = self.env.reset()
        done = False
        ep_len = 0

        while not done:
            # Collect one transition
            action, log_prob, state_value = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Bootstrap if truncated (???)
            # if truncated:
            #     _, _, next_state_value = self.agent.get_action(next_state)
            #     reward += self.gamma * next_state_value.detach()

            self.buffer.add(state, action, reward, next_state, done, log_prob, state_value)
            ep_len += 1

            if not done:
                state = next_state

        return ep_len


    def train(self):
        train_stats = {
            'episode_length': [],
            'episode_return': []
        }
        self.buffer.clear()

        filename = self.params.get('save_filename', None)

        # ep_bar = tqdm.trange(self.params['num_episodes'], position=0, leave=True)
        for ep in range(self.params['num_episodes']):
            ep_len = self.rollout()
            train_stats['episode_length'].append(ep_len)

            # Train every episode
            update_stats = self.ppo_update()
            update_parent_info(train_stats, update_stats, extend=True)

            # Compute moving average of rewards to track training progress
            avg_window = self.params['avg_return_window']
            if len(train_stats['episode_return']) > avg_window:
                avg_return = np.mean(train_stats['episode_return'][-avg_window:])
            else:
                avg_return = np.mean(train_stats['episode_return'])

            # Logging for tracking progress
            G = train_stats['episode_return'][-1]
            loss = train_stats['combined_loss'][-1]

            if ep % self.params['log_frequency'] == 0 and ep > 0:
                print(f"Episode: {ep} | Return: {G} | Moving Avg Return: {avg_return} | Loss: {loss:.3f}")

            # ep_bar.set_description(f"Episode: {ep} | Return: {G} | Moving Avg Return: {avg_return} | Loss: {loss:.3f}")

            # Saving model checkpoints
            if filename is not None and ep % self.params['model_save_frequency'] == 0 and ep > 0:
                self.agent.save_model(f'{filename}-ep_{ep}.pth')

        # Training is complete
        # Save final model
        if filename is not None:
            self.agent.save_model(f'{filename}-ep_{ep}.pth')

        # Return training statistics
        return train_stats

# %% [markdown]
# ## Train PPO on Lunar Lander

# %%
faulthandler.enable()

def train_ppo_lunar():
    env = gym.make('LunarLander-v3')

    train_params = {
        # env info
        'state_dim': 8,

        # training params
        'num_episodes': 3000,
        'num_trials': 1,
        'buffer_size': 2_000,
        # 'update_frequency': 1, # 10_000,
        # 'warmup_period': 10_000,

        # hyperparams
        'actor_learning_rate': 3e-4, # 3e-4,
        'critic_learning_rate': 1e-3, #1e-3,
        'gamma': 0.99,
        'entropy_coef': 0, # 0.01,
        'clip_epsilon': 0.2, # ppo clip constraint
        'epochs_per_update': 5,
        'save_filename': './ppo-lunar-lander',
        'model_save_frequency': 750, # save checkpoint every 250 episodes

        # Stats tracking parameters
        'avg_return_window': 25,
        'log_frequency': 50
    }

    trial_stats = {}

    for trial in range(train_params['num_trials']):
        print(f'Trial {trial}:')

        my_agent = PPOAgent(state_dim=train_params['state_dim'],
                            action_dim=4,
                            hidden_dim=128,
                            num_hidden=2)

        my_trainer = PPOAgentTrainer(my_agent, env, train_params)
        train_stats = my_trainer.train()
        update_parent_info(trial_stats, train_stats)

    return trial_stats


lunar_stats = train_ppo_lunar()
save_object('./ppo-lunar-stats.pkl', lunar_stats)

plot_curves([np.stack(lunar_stats['episode_return'])],
            ['PPO'],
            ['r'],
            xlabel='Episodes',
            ylabel='Discounted return (gamma=0.99)',
            fig_title='Training Return, Vanilla PPO Lunar Lander',
            smoothing=True)

plot_curves([np.stack(lunar_stats['actor_loss'])],
            ['PPO'],
            ['g'],
            xlabel='Episodes',
            ylabel='Loss',
            fig_title='Actor Loss, Vanilla PPO Lunar Lander',
            smoothing=True)

plot_curves([np.stack(lunar_stats['critic_loss'])],
            ['PPO'],
            ['b'],
            xlabel='Episodes',
            ylabel='Loss',
            fig_title='Critic Loss, Vanilla PPO Lunar Lander',
            smoothing=True)

# %%
print(lunar_stats.keys())

# %%


# %%



