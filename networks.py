# networks.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# config needs to be imported to get LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS
import config

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, name="actor", model_name=config.MODEL_NAME, checkpoints_dir=config.CHECKPOINT_DIR):
        super(ActorNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_file = os.path.join(self.checkpoints_dir, self.model_name, name + ".pth")
        os.makedirs(os.path.dirname(self.checkpoints_file), exist_ok=True)

        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.lstm_num_layers = config.LSTM_NUM_LAYERS

        # Layers before LSTM
        self.fc1 = nn.Linear(input_dims, 256)
        # LSTM layer
        self.lstm = nn.LSTM(256, self.lstm_hidden_size, self.lstm_num_layers, batch_first=True)
        # Layers after LSTM
        self.fc2 = nn.Linear(self.lstm_hidden_size, 256)
        self.pi = nn.Linear(256, n_actions)

    def forward(self, state, hidden_in, cell_in):
        # state shape: (batch_size, sequence_len, input_dims)
        # For single step processing (sequence_len=1): (batch_size, 1, input_dims)
        # hidden_in, cell_in shape: (num_layers, batch_size, lstm_hidden_size)

        x = F.relu(self.fc1(state))

        # If state is (batch_size, input_dims), need to add sequence_len dimension
        if x.ndim == 2:
            x = x.unsqueeze(1) # -> (batch_size, 1, features_after_fc1)

        # LSTM expects input: (batch, seq_len, input_size)
        # LSTM returns output: (batch, seq_len, hidden_size), (h_n, c_n)
        lstm_out, (hidden_out, cell_out) = self.lstm(x, (hidden_in, cell_in))

        # If processing single step, lstm_out is (batch, 1, hidden_size). Squeeze seq_len dim.
        if lstm_out.size(1) == 1:
            lstm_out = lstm_out.squeeze(1)
        else: # if processing sequence, take the last output of the sequence
            lstm_out = lstm_out[:, -1, :]


        x = F.relu(self.fc2(lstm_out))
        pi = torch.tanh(self.pi(x)) # Output actions between -1 and 1
        return pi, hidden_out, cell_out

    def save_checkpoint(self): # No change
        os.makedirs(os.path.dirname(self.checkpoints_file), exist_ok=True)
        print(f"... saving checkpoint '{self.checkpoints_file}' ...")
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self): # No change
        print(f"... loading checkpoint '{self.checkpoints_file}' ...")
        self.load_state_dict(torch.load(self.checkpoints_file, map_location=config.DEVICE))
        self.to(config.DEVICE)


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, name="critic", model_name=config.MODEL_NAME, checkpoints_dir=config.CHECKPOINT_DIR):
        super(CriticNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_file = os.path.join(self.checkpoints_dir, self.model_name, name + ".pth")
        os.makedirs(os.path.dirname(self.checkpoints_file), exist_ok=True)

        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.lstm_num_layers = config.LSTM_NUM_LAYERS

        # Input to first layer includes state and action dimensions
        # For RDPG, action is typically concatenated AFTER LSTM processes state, or state+action fed to LSTM
        # Let's try state -> LSTM -> concat(LSTM_out, action) -> FC layers
        self.fc1_state = nn.Linear(input_dims, 128) # Process state first
        self.lstm = nn.LSTM(128, self.lstm_hidden_size, self.lstm_num_layers, batch_first=True)

        # Layers after LSTM and action concatenation
        self.fc1_action = nn.Linear(n_actions, 128) # Process action separately
        self.fc2 = nn.Linear(self.lstm_hidden_size + 128, 256) # Concat LSTM_out and processed_action
        self.q = nn.Linear(256, 1)

    def forward(self, state, action, hidden_in, cell_in):
        # state shape: (batch_size, sequence_len, input_dims)
        # action shape: (batch_size, sequence_len, n_actions) or (batch_size, n_actions)
        # hidden_in, cell_in shape: (num_layers, batch_size, lstm_hidden_size)

        state_processed = F.relu(self.fc1_state(state))
        if state_processed.ndim == 2: # Add sequence dim if not present
            state_processed = state_processed.unsqueeze(1)

        lstm_out, (hidden_out, cell_out) = self.lstm(state_processed, (hidden_in, cell_in))

        if lstm_out.size(1) == 1: # Single step processing
            lstm_out_squeezed = lstm_out.squeeze(1)
        else: # Sequence processing, take last output
            lstm_out_squeezed = lstm_out[:, -1, :]

        action_processed = F.relu(self.fc1_action(action))
        # If action was (batch, n_actions), it's fine
        # If action was (batch, seq_len, n_actions) and seq_len=1, squeeze it
        if action_processed.ndim == 3 and action_processed.size(1) == 1:
             action_processed = action_processed.squeeze(1)


        x = torch.cat([lstm_out_squeezed, action_processed], dim=1)
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q, hidden_out, cell_out

    def save_checkpoint(self): # No change
        os.makedirs(os.path.dirname(self.checkpoints_file), exist_ok=True)
        print(f"... saving checkpoint '{self.checkpoints_file}' ...")
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self): # No change
        print(f"... loading checkpoint '{self.checkpoints_file}' ...")
        self.load_state_dict(torch.load(self.checkpoints_file, map_location=config.DEVICE))
        self.to(config.DEVICE)