# train.py
import gymnasium as gym
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # If needed
from collections import deque
import matplotlib.pyplot as plt
import traceback
import time
import torch

import config
import custom_panda_env # Registers custom env (now with partial obs logic)
from rdpg_agent import RDPGAgent # <--- Use RDPGAgent
try:
    from evaluate import evaluate_agent # Keep preprocess_observation local or from utils
except ImportError:
    print("Warning: Could not import 'evaluate_agent' from evaluate.py.")
    def evaluate_agent(*args, **kwargs): print("Placeholder: evaluate_agent not imported."); return None

# --- Preprocessing Function --- (Must match agent's input_dims expectation)
def preprocess_observation(obs_dict):
    """Concatenates components of the observation dictionary into a flat vector."""
    if not isinstance(obs_dict, dict):
        print(f"Warning: preprocess_observation expected dict, got {type(obs_dict)}")
        if isinstance(obs_dict, np.ndarray): return obs_dict.flatten().astype(np.float32)
        return None
    # Agent's state = robot's own partial observation + current desired goal
    robot_state_vec = obs_dict.get('observation', np.array([])) # Should be (6,) from custom env
    desired_goal_vec = obs_dict.get('desired_goal', np.array([])) # Should be (3,)
    state_parts = [robot_state_vec.flatten(), desired_goal_vec.flatten()]
    try:
        valid_parts = [part for part in state_parts if isinstance(part, np.ndarray) and part.size > 0]
        if len(valid_parts) < 2 : # Expecting both parts
            print(f"Warning: Preprocess_observation missing components. Robot_obs size: {robot_state_vec.size}, Desired_goal size: {desired_goal_vec.size}")
            return None
        flat_obs = np.concatenate(valid_parts)
        return flat_obs.astype(np.float32)
    except ValueError as e: print(f"Error in preprocess_observation: {e}"); return None


if __name__ == "__main__":
    # os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # If needed

    print(f"Using device: {config.DEVICE}") # ... (other version prints) ...
    # ... (PyTorch/CUDA version prints) ...

    env = None
    try:
        print(f"Creating training environment '{config.ENV_NAME}' (partial obs)...")
        env = gym.make(config.ENV_NAME, render_mode="rgb_array")
        print("Training environment created.")
        if isinstance(env.observation_space, gym.spaces.Dict) and 'observation' in env.observation_space.spaces:
             print(f"Verified Robot PARTIAL Observation Space Shape from Env: {env.observation_space['observation'].shape}")
             print(f"Verified Desired Goal Space Shape from Env: {env.observation_space['desired_goal'].shape}")
        else: print(f"Warning: Could not verify obs space structure. Type: {type(env.observation_space)}")
    except Exception as e: print(f"Fatal Error creating env: {e}"); traceback.print_exc(); exit()

    agent = None
    try:
        print("Initializing RDPG agent...")
        agent = RDPGAgent(env=env) # <--- Use RDPGAgent
        print("RDPG Agent initialized.")
    except Exception as e:
        print(f"Fatal Error initializing agent: {e}"); traceback.print_exc(); 
        if env:env.close(); exit()

    best_avg_score = -np.inf; score_history = deque(maxlen=100); avg_score_history = []
    print(f"Starting training for {config.N_GAMES} episodes on {config.ENV_NAME}...") # ... (other config prints)
    total_steps = 0

    for i_episode in range(config.N_GAMES):
        start_time_episode = time.time()
        observation_dict, info = env.reset()
        current_flat_state = preprocess_observation(observation_dict)
        if current_flat_state is None: print(f"Ep {i_episode+1} skip: bad initial obs"); continue

        agent.reset_actor_hidden_state() # <--- RESET AGENT'S HIDDEN STATE FOR NEW EPISODE

        terminated, truncated = False, False
        episode_score, episode_steps = 0, 0
        time_action_total,time_step_total,time_remember_total,time_learn_total,learn_calls = 0,0,0,0,0

        while not (terminated or truncated):
            if current_flat_state is None: break

            # --- Store current actor hidden state BEFORE action selection for replay buffer ---
            # These are (num_layers, 1, hidden_size) from agent's internal state
            h_actor_for_buffer = agent.actor_h.detach().clone()
            c_actor_for_buffer = agent.actor_c.detach().clone()
            # -----------------------------------------------------------------------------

            time_action_start = time.time()
            if total_steps < config.WARMUP_BATCHES * config.BATCH_SIZE:
                action = env.action_space.sample()
                # During warmup with random actions, we still need to step the agent's hidden state
                # if we want the stored hidden states to be meaningful.
                # For simplicity, let's pass the current_flat_state through actor to update hidden state,
                # but discard the action. This keeps hidden state flowing.
                if agent.actor_h is not None and agent.actor_c is not None : # Ensure hidden states are init
                    _ = agent.choose_action(current_flat_state, evaluate=True) # Call to update hidden state, ignore action
            else:
                if current_flat_state.shape[0] != agent.input_dims:
                    print(f"Error: State shape mismatch! {current_flat_state.shape} vs {agent.input_dims}. Truncating."); truncated=True; continue
                action = agent.choose_action(current_flat_state, evaluate=False) # This updates agent.actor_h/c
            time_action_end = time.time(); time_action_total += (time_action_end - time_action_start)

            next_flat_state = None; reward = 0.0
            time_step_start = time.time()
            try:
                next_observation_dict, reward, terminated, truncated, info = env.step(action)
                truncated = truncated or (episode_steps + 1 >= config.MAX_EPISODE_STEPS)
                next_flat_state = preprocess_observation(next_observation_dict)
                if next_flat_state is None: truncated = True
            except Exception as e: print(f"Error in env.step: {e}"); traceback.print_exc(); truncated=True; reward=-10.0
            time_step_end = time.time(); time_step_total += (time_step_end - time_step_start)

            time_remember_start = time.time()
            if current_flat_state is not None and next_flat_state is not None:
                 if next_flat_state.shape[0] == agent.input_dims:
                      # Pass h_actor_for_buffer, c_actor_for_buffer (hidden state *before* action)
                      agent.remember(current_flat_state, action, reward, next_flat_state,
                                     terminated or truncated, h_actor_for_buffer, c_actor_for_buffer)
                 else: print(f"Warning: Skip remember due to invalid next_state shape {next_flat_state.shape}"); truncated=True
            elif current_flat_state is not None and next_flat_state is None:
                 print(f"Info: Skip remember due to invalid next_state.")
            time_remember_end = time.time(); time_remember_total += (time_remember_end - time_remember_start)

            # If episode ended, reset actor hidden state for the *next* episode's first step.
            # This reset is done at the top of the episode loop.
            # However, for the *current transition being stored*, if 'done' is true,
            # the 'next_hidden_state' is effectively a reset state for the LSTM in training.
            # The replay buffer design currently stores h_in for (s), not h_out for (s').

            time_learn_start = time.time(); learn_executed_this_step = False
            if total_steps >= config.WARMUP_BATCHES * config.BATCH_SIZE and len(agent.memory) >= config.BATCH_SIZE:
                 agent.learn(); learn_executed_this_step = True; learn_calls += 1
            time_learn_end = time.time()
            if learn_executed_this_step: time_learn_total += (time_learn_end - time_learn_start)

            current_flat_state = next_flat_state
            episode_score += reward; episode_steps += 1; total_steps += 1

            if total_steps % 200 == 0 and episode_steps > 0: # Periodic Timing Print
                 avg_act=time_action_total/episode_steps; avg_step=time_step_total/episode_steps; avg_rem=time_remember_total/episode_steps; avg_learn=(time_learn_total/learn_calls) if learn_calls > 0 else 0
                 print(f"      Timing Avg (Ep {i_episode+1}, {episode_steps} steps): Act={avg_act:.5f}s | EnvStep={avg_step:.5f}s | Rem={avg_rem:.5f}s | Learn={avg_learn:.5f}s")

        # ... (End of Episode: score logging, model saving, periodic eval as before) ...
        end_time_episode = time.time(); episode_duration = end_time_episode - start_time_episode
        score_history.append(episode_score); avg_score = np.mean(score_history); avg_score_history.append(avg_score)
        if len(score_history) >= 100 and avg_score > best_avg_score : best_avg_score = avg_score; agent.save_models(); print(f"*** New best avg score: {best_avg_score:.2f} ***")
        print(f"Ep {i_episode+1}/{config.N_GAMES} | Steps: {episode_steps} | Score: {episode_score:.2f} | Avg Score: {avg_score:.2f} | Buffer: {len(agent.memory)} | Ep Time: {episode_duration:.2f}s")
        if (i_episode + 1) % config.EVAL_FREQ == 0 and i_episode > 0 and 'evaluate_agent' in globals():
             print(f"\n--- Periodic Eval ---"); eval_metrics = evaluate_agent(agent, config.ENV_NAME, n_episodes=3, render_mode="none"); print(f"--- Eval Result --- Avg Score: {eval_metrics['avg_score']:.2f}, Avg MSE: {eval_metrics.get('avg_mse', float('nan')):.6f}" if eval_metrics else "Eval failed."); print(f"--- Resuming Training ---\n")


    if env: env.close(); print("Training environment closed.")
    print("Training finished.")
    # ... (Plotting as before) ...
    if avg_score_history: plt.figure(figsize=(12,6)); plt.plot(list(range(1,len(avg_score_history)+1)), avg_score_history); plt.xlabel("Episode"); plt.ylabel("Avg Score (100 ep)"); plt.title(f"Training Progress {config.MODEL_NAME}"); plt.grid(True); plt.savefig(f"{config.MODEL_NAME}_train_score.png"); print("Score plot saved."); # plt.show()