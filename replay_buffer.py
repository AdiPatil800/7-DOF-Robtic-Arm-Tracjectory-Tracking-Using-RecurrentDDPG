# replay_buffer.py
import numpy as np
import torch
import config

class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.size = max_size
        self.counter = 0
        self.input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        self.n_actions = n_actions
        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.lstm_num_layers = config.LSTM_NUM_LAYERS

        self.states = np.zeros((max_size, *self.input_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.new_states = np.zeros((max_size, *self.input_shape), dtype=np.float32)
        self.terminals = np.zeros(max_size, dtype=bool)

        # Store actor's LSTM hidden and cell states
        self.actor_h_in = np.zeros((max_size, self.lstm_num_layers, self.lstm_hidden_size), dtype=np.float32)
        self.actor_c_in = np.zeros((max_size, self.lstm_num_layers, self.lstm_hidden_size), dtype=np.float32)

        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_priority = 1.0
        self.alpha = config.PER_ALPHA
        self.beta = config.PER_BETA_START
        if config.PER_BETA_FRAMES > 0: self.beta_increment = (1.0 - config.PER_BETA_START) / config.PER_BETA_FRAMES
        else: self.beta_increment = 0

    def store_transition(self, state, action, reward, new_state, done, h_actor, c_actor):
        index = self.counter % self.size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.terminals[index] = done
        self.actor_h_in[index] = h_actor
        self.actor_c_in[index] = c_actor
        self.priorities[index] = self.max_priority
        self.counter += 1

    def _get_probabilities(self):
        current_size = self.current_buffer_size() # <-- Corrected line
        if current_size == 0:
            return np.array([], dtype=np.float32)
        priorities_to_scale = self.priorities[:current_size]
        scaled_priorities = np.power(priorities_to_scale + 1e-9, self.alpha) # Added epsilon for stability
        sum_scaled_priorities = np.sum(scaled_priorities)
        if sum_scaled_priorities == 0:
             print("Warning: Sum of scaled priorities is zero in _get_probabilities. Using uniform sampling.")
             return np.full(current_size, 1.0 / current_size, dtype=np.float32)
        return scaled_priorities / sum_scaled_priorities

    def _get_importance_sampling_weights(self, probabilities, indices):
        current_size = self.current_buffer_size() # <-- Corrected line
        if current_size == 0 or len(indices) == 0:
            return np.array([], dtype=np.float32)

        self.beta = min(1.0, self.beta + self.beta_increment)
        probs_for_indices = probabilities[indices]
        probs_for_indices = np.maximum(probs_for_indices, 1e-9) # Avoid division by zero

        weights = np.power(current_size * probs_for_indices, -self.beta)
        
        # Normalize weights by the maximum possible weight
        min_prob_overall = np.min(probabilities[probabilities > 0]) if np.any(probabilities > 0) else 1e-9
        min_prob_overall = max(min_prob_overall, 1e-9) # Ensure not zero
        max_possible_weight = np.power(current_size * min_prob_overall, -self.beta)

        if max_possible_weight > 0:
            weights /= max_possible_weight
        else:
             print("Warning: max_possible_weight calculation resulted in zero in _get_importance_sampling_weights.")
             weights.fill(1.0) # Fallback
        return weights

    def sample(self, batch_size):
        max_mem = self.current_buffer_size()
        if max_mem < batch_size:
             # print(f"Buffer size {max_mem} < batch_size {batch_size}. Cannot sample.")
             return None

        probabilities = self._get_probabilities()
        if len(probabilities) == 0 or max_mem == 0: # Ensure probabilities array is not empty
             # print("Warning: Probabilities array is empty or buffer is empty. Cannot sample.")
             return None
        
        # Ensure probabilities sum to 1, handle potential floating point inaccuracies
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0):
            if prob_sum > 0 : # Only normalize if sum is positive
                # print(f"Warning: Probabilities do not sum to 1 ({prob_sum}). Normalizing.")
                probabilities /= prob_sum
            else: # If sum is zero (e.g., all priorities were zero and scaled to zero)
                print("Warning: Probabilities sum to zero. Using uniform sampling for this batch.")
                probabilities = np.full(max_mem, 1.0 / max_mem, dtype=np.float32)


        try:
            indices = np.random.choice(max_mem, batch_size, p=probabilities, replace=True)
        except ValueError as e:
            print(f"Error in np.random.choice: {e}. Probabilities sum: {np.sum(probabilities)}, Length: {len(probabilities)}, Max_mem: {max_mem}")
            # Fallback: uniform sampling if p is problematic
            indices = np.random.choice(max_mem, batch_size, replace=True)


        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        new_states = self.new_states[indices]
        dones = self.terminals[indices]
        h_actors = self.actor_h_in[indices]
        c_actors = self.actor_c_in[indices]
        weights = self._get_importance_sampling_weights(probabilities, indices)

        # Convert to PyTorch tensors
        states_t = torch.tensor(states, dtype=torch.float32).to(config.DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(config.DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(config.DEVICE)
        new_states_t = torch.tensor(new_states, dtype=torch.float32).to(config.DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(config.DEVICE)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)
        h_actors_t = torch.tensor(h_actors, dtype=torch.float32).to(config.DEVICE)
        c_actors_t = torch.tensor(c_actors, dtype=torch.float32).to(config.DEVICE)

        # Reshape hidden states for LSTM: (num_layers, batch_size, hidden_size)
        h_actors_t = h_actors_t.permute(1, 0, 2)
        c_actors_t = c_actors_t.permute(1, 0, 2)

        return states_t, actions_t, rewards_t, new_states_t, dones_t, \
               h_actors_t, c_actors_t, indices, weights_t

    def update_priorities(self, indices, td_errors_numpy): # Expects numpy array of abs TD errors
        priorities = td_errors_numpy + config.PER_EPSILON # Already abs value from agent
        clipped_priorities = np.minimum(priorities, self.max_priority)

        valid_mask = indices < self.current_buffer_size()
        valid_indices = indices[valid_mask]
        valid_clipped_priorities = clipped_priorities[valid_mask]

        if len(valid_indices) > 0:
             self.priorities[valid_indices] = valid_clipped_priorities
             current_max_in_batch = np.max(valid_clipped_priorities)
             if current_max_in_batch > self.max_priority:
                  self.max_priority = current_max_in_batch

    def current_buffer_size(self): return min(self.size, self.counter)
    def __len__(self): return self.current_buffer_size()