# config.py
import torch
import numpy as np

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment ---
ENV_NAME = 'PandaSinusoidTrack-v0' # Custom environment
# Max steps defined by environment registration

# --- Trajectory Parameters (used by custom_panda_env.py) ---
TRAJ_CENTER = np.array([0.3, 0.0, 0.2])
TRAJ_RADIUS = 0.15
TRAJ_FREQ = 0.1 # Hz
TRAJ_Z_AMP = 0.05
TRAJ_DURATION = 10.0 # seconds
CONTROL_FREQ = 20 # Hz
MAX_EPISODE_STEPS = int(TRAJ_DURATION * CONTROL_FREQ)

# --- RDPG Agent (was DDPG) ---
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005
NOISE_STDDEV = 0.1
NOISE_CLIP = 0.5
POLICY_DELAY = 2
OPTIMIZER_WEIGHT_DECAY = 0.0
LSTM_HIDDEN_SIZE = 128 # <--- NEW: Size of LSTM hidden layer
LSTM_NUM_LAYERS = 1    # <--- NEW: Number of LSTM layers

# --- Replay Buffer ---
BUFFER_SIZE = int(1e6) # May need adjustment for storing hidden states
BATCH_SIZE = 64      # <--- REDUCED BATCH SIZE (RNNs are more memory intensive)
# Note: True RDPG often uses sequence replay, our buffer is a simpler transition replay
# with hidden states. Consider reducing buffer size if memory is an issue.

# --- Prioritized Replay (PER) ---
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = int(1e6)
PER_EPSILON = 1e-6

# --- Training ---
N_GAMES = 3000 # Adjust as needed
WARMUP_BATCHES = 100 # Steps_random_action = WARMUP_BATCHES * BATCH_SIZE
EVAL_FREQ = 100
CHECKPOINT_DIR = "checkpoints"
MODEL_NAME = "rdpg_traj_track" # <--- New model name for RDPG

# --- Evaluation ---
EVAL_EPISODES = 10
RENDER_DELAY = 50
RENDER_FILENAME = f"{MODEL_NAME}_evaluation.gif"