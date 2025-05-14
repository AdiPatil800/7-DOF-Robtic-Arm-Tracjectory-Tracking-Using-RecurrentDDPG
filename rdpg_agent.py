# rdpg_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import gymnasium as gym

import config
from networks import ActorNetwork, CriticNetwork # Networks are recurrent
from replay_buffer import PrioritizedReplayBuffer # Buffer stores hidden states

class RDPGAgent:
    def __init__(self, env):
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.policy_delay = config.POLICY_DELAY
        self.noise_stddev = config.NOISE_STDDEV
        self.noise_clip = config.NOISE_CLIP
        self.learn_step_counter = 0

        self.env = env # Keep env reference if needed

        # --- Determine input_dims from the Dict observation space ---
        if not isinstance(env.observation_space, gym.spaces.Dict):
             raise ValueError(f"Agent expects a Dict observation space, but got {type(env.observation_space)}")
        try:
             # This is the robot's own observation (e.g., EE pose, (6,) based on debug)
             robot_obs_dim = env.observation_space['observation'].shape[0]
             # This is the target trajectory point (e.g., (3,))
             goal_dim = env.observation_space['desired_goal'].shape[0]

             # Agent's input = robot_observation_vector + desired_goal_vector
             self.input_dims = robot_obs_dim + goal_dim
             print(f"RDPG Agent calculated input_dims: {self.input_dims} (robot_obs_part={robot_obs_dim}, desired_goal={goal_dim})")

             if self.input_dims <= 0: # Check for non-positive dimensions
                  raise ValueError(f"Calculated input_dims ({self.input_dims}) is not positive. Check observation space and preprocess_observation logic.")
        except KeyError as e:
             raise ValueError(f"Observation space Dict missing expected key for input_dims: {e}. Keys: {list(env.observation_space.spaces.keys())}")
        except AttributeError as e:
             raise ValueError(f"Error accessing shape from observation space components: {e}")
        except Exception as e:
             raise ValueError(f"Unexpected error calculating input_dims: {e}")
        # --------------------------------------------------------------------------
        self.n_actions = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        if self.action_scale == 0: # Avoid division by zero for constant action spaces
            print("Warning: Action scale is zero. Setting to 1.0, bias to 0.0.")
            self.action_scale = 1.0
            self.action_bias = 0.0

        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.lstm_num_layers = config.LSTM_NUM_LAYERS

        # --- Initialize Actor's hidden state for current episode ---
        self.actor_h = None
        self.actor_c = None
        self.reset_actor_hidden_state() # Initialize hidden states
        # -----------------------------------------------------------

        self.memory = PrioritizedReplayBuffer(config.BUFFER_SIZE, self.input_dims, self.n_actions)
        self._initialize_networks()


    def _initialize_networks(self): # Networks are now recurrent
        model_name = config.MODEL_NAME
        self.actor = ActorNetwork(self.input_dims, self.n_actions, name="actor", model_name=model_name).to(config.DEVICE)
        self.critic = CriticNetwork(self.input_dims, self.n_actions, name="critic", model_name=model_name).to(config.DEVICE)
        self.target_actor = ActorNetwork(self.input_dims, self.n_actions, name="target_actor", model_name=model_name).to(config.DEVICE)
        self.target_critic = CriticNetwork(self.input_dims, self.n_actions, name="target_critic", model_name=model_name).to(config.DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)

        self.update_parameters(tau=1.0) # Initialize target networks
        self.prep_for_eval() # Set target nets to eval mode initially

    def reset_actor_hidden_state(self, batch_size_override=None):
        """Resets the actor's LSTM hidden and cell states to zeros."""
        current_batch_size = batch_size_override if batch_size_override is not None else 1
        # For action selection, batch_size is 1. For LSTM layer, needs (num_layers, batch_size, hidden_size)
        self.actor_h = torch.zeros(self.lstm_num_layers, current_batch_size, self.lstm_hidden_size, device=config.DEVICE).detach()
        self.actor_c = torch.zeros(self.lstm_num_layers, current_batch_size, self.lstm_hidden_size, device=config.DEVICE).detach()


    def choose_action(self, state, evaluate=False): # state is already preprocessed flat vector
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        if state.ndim > 1: state = state.flatten()
        if state.shape[0] != self.input_dims:
            print(f"CRITICAL Error in choose_action: State shape {state.shape} != input_dims ({self.input_dims}). Returning zero action.")
            return np.zeros(self.n_actions) # Fallback

        # Reshape state for LSTM: (batch_size=1, seq_len=1, features)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)

        self.actor.eval() # Set network to eval mode for action selection
        with torch.no_grad():
            # Pass current hidden states to actor
            mu, new_h, new_c = self.actor(state_tensor, self.actor_h, self.actor_c)
        # No need to set actor.train() here if only used for inference, learn() handles train mode

        # Update agent's recurrent state for the next step
        self.actor_h = new_h.detach()
        self.actor_c = new_c.detach()

        action_from_policy = mu.squeeze(0).squeeze(0) # Remove batch and seq_len dims

        if not evaluate:
            noise = torch.normal(mean=0.0, std=self.noise_stddev, size=action_from_policy.shape).to(config.DEVICE)
            action_from_policy = torch.clamp(action_from_policy + noise, -1.0, 1.0)

        action_scaled = action_from_policy.cpu().numpy() * self.action_scale + self.action_bias
        action_clipped = np.clip(action_scaled, self.min_action, self.max_action)
        return action_clipped

    # h_actor_to_store, c_actor_to_store are the hidden states *before* the action was chosen
    def remember(self, state, action, reward, new_state, done, h_actor_to_store, c_actor_to_store):
        if not isinstance(action, np.ndarray): action = np.array(action)
        unscaled_action = np.clip((action - self.action_bias) / self.action_scale if self.action_scale != 0 else action, -1.0, 1.0)

        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        if not isinstance(new_state, np.ndarray): new_state = np.array(new_state, dtype=np.float32)
        if state.ndim > 1: state = state.flatten()
        if new_state.ndim > 1: new_state = new_state.flatten()
        if state.shape[0] != self.input_dims or new_state.shape[0] != self.input_dims:
             print(f"Error in remember: State shapes mismatch. State: {state.shape}, New: {new_state.shape}, Expected: ({self.input_dims},). Skipping."); return

        # Convert hidden states to numpy for storage.
        # h_actor_to_store/c_actor_to_store are (num_layers, 1, hidden_size) from agent's state
        h_np = h_actor_to_store.squeeze(1).cpu().numpy() # Remove batch_size=1 dim
        c_np = c_actor_to_store.squeeze(1).cpu().numpy() # Remove batch_size=1 dim

        self.memory.store_transition(state, unscaled_action, reward, new_state, done, h_np, c_np)


    def learn(self):
        if len(self.memory) < self.batch_size: return
        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None: return

        states, actions, rewards, new_states, dones, \
        h_actor_in_batch, c_actor_in_batch, indices, weights = sample_result
        # h_actor_in_batch, c_actor_in_batch are already permuted: (num_layers, batch_size, hidden_size)

        # Reshape states for LSTM: (batch_size, seq_len=1, features)
        states_lstm = states.unsqueeze(1)
        new_states_lstm = new_states.unsqueeze(1)
        actions_lstm = actions.unsqueeze(1) # For critic if it processes actions sequentially via LSTM

        rewards = rewards.unsqueeze(1)
        dones_float = dones.unsqueeze(1).type(torch.float32)
        weights = weights.unsqueeze(1)

        # Zero initial hidden states for critic (simplification for single transition replay)
        h_critic_zero = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size, device=config.DEVICE).detach()
        c_critic_zero = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size, device=config.DEVICE).detach()

        self.target_actor.eval(); self.target_critic.eval(); self.critic.train(); self.actor.train()

        with torch.no_grad():
            # Use h_actor_in_batch (hidden state for 'states') as input to target_actor for 'new_states'
            # This is an approximation: assumes hidden state doesn't change much, or that we are
            # learning value of (s,h_s,a) and next state value given (s',h_s).
            # A more complex approach would recompute h for new_states based on a sequence.
            next_actions, _, _ = self.target_actor(new_states_lstm, h_actor_in_batch, c_actor_in_batch)
            noise = torch.normal(mean=0.0, std=self.noise_stddev, size=next_actions.shape).to(config.DEVICE)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)

            target_q_values, _, _ = self.target_critic(new_states_lstm, next_actions, h_critic_zero, c_critic_zero)
            target_q = rewards + (self.gamma * target_q_values * (1.0 - dones_float))

        current_q_values, _, _ = self.critic(states_lstm, actions_lstm, h_critic_zero, c_critic_zero)
        td_errors_tensor = (target_q - current_q_values).abs() # Tensor for priorities
        critic_loss = (weights * F.mse_loss(current_q_values, target_q, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.memory.update_priorities(indices, td_errors_tensor.squeeze().detach().cpu().numpy())

        self.learn_step_counter += 1
        if self.learn_step_counter % self.policy_delay == 0:
            for param in self.critic.parameters(): param.requires_grad = False
            # Actor takes 'states' and the hidden states that were input to get 'actions'
            current_actions_pi, _, _ = self.actor(states_lstm, h_actor_in_batch, c_actor_in_batch)
            actor_q_values, _, _ = self.critic(states_lstm, current_actions_pi, h_critic_zero, c_critic_zero)
            actor_loss = -actor_q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param in self.critic.parameters(): param.requires_grad = True
            self.update_parameters()

    def update_parameters(self, tau=None):
        if tau is None: tau = self.tau
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()): tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()): tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def save_models(self):
        print(f"---- saving RDPG models for {config.MODEL_NAME} ----"); self.actor.save_checkpoint(); self.critic.save_checkpoint(); self.target_actor.save_checkpoint(); self.target_critic.save_checkpoint()
    def load_models(self):
        print(f"---- loading RDPG models for {config.MODEL_NAME} ----"); self.actor.load_checkpoint(); self.critic.load_checkpoint(); self.target_actor.load_checkpoint(); self.target_critic.load_checkpoint(); self.prep_for_eval()
    def prep_for_eval(self):
        self.actor.eval(); self.critic.eval(); self.target_actor.eval(); self.target_critic.eval()