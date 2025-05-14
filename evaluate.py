# evaluate.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # If needed
import gymnasium as gym
import numpy as np
import imageio; import time; import traceback; import matplotlib.pyplot as plt; import torch

import config
import custom_panda_env # Registers custom env (now with partial obs)
from rdpg_agent import RDPGAgent # <--- Import RDPGAgent

# --- Preprocessing Function --- (Must match train.py and agent's input_dims expectation)
def preprocess_observation(obs_dict):
    """Concatenates components of the observation dictionary into a flat vector."""
    if not isinstance(obs_dict, dict):
        print(f"Warning: preprocess_observation expected dict, got {type(obs_dict)}")
        if isinstance(obs_dict, np.ndarray): return obs_dict.flatten().astype(np.float32)
        return None
    robot_state_vec = obs_dict.get('observation', np.array([])) # Should be (6,)
    desired_goal_vec = obs_dict.get('desired_goal', np.array([])) # Should be (3,)
    state_parts = [robot_state_vec.flatten(), desired_goal_vec.flatten()]
    try:
        valid_parts = [part for part in state_parts if isinstance(part, np.ndarray) and part.size > 0]
        if len(valid_parts) < 2:
            print(f"Warning: Preprocess_observation missing components. Robot_obs size: {robot_state_vec.size}, Desired_goal size: {desired_goal_vec.size}")
            return None
        flat_obs = np.concatenate(valid_parts)
        return flat_obs.astype(np.float32)
    except ValueError as e: print(f"Error in preprocess_observation: {e}"); return None

def evaluate_agent(agent, env_name, n_episodes=10, render_mode="human", delay=0.03):
    # ... (print start, env creation with n_actions as before) ...
    print(f"\n--- Starting Evaluation ({render_mode} mode) ---") # ... etc.
    if render_mode not in ["human", "rgb_array", "none"]: render_mode = "none"
    env = None; n_actions = None
    try:
        create_render_mode = render_mode if render_mode != "none" else "rgb_array"
        env = gym.make(env_name, render_mode=create_render_mode)
        n_actions = env.action_space.shape[0]
    except Exception as e: print(f"Error creating eval env: {e}"); traceback.print_exc(); return None

    agent.prep_for_eval()
    total_rewards, episode_lengths, all_episode_mse, frames = [], [], [], []
    plot_dir = f"eval_plots_{config.MODEL_NAME}"; os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving evaluation plots to: {plot_dir}")

    for i in range(n_episodes):
        print(f"--- Starting Eval Episode {i+1}/{n_episodes} ---")
        observation_dict, info = env.reset()
        current_flat_state = preprocess_observation(observation_dict)
        if current_flat_state is None: print(f"Eval ep {i+1} skip: bad initial obs"); continue

        agent.reset_actor_hidden_state() # <--- RESET AGENT'S HIDDEN STATE FOR EVAL EPISODE

        terminated, truncated, episode_reward, steps = False, False, 0, 0
        actual_positions, ref_positions, actions_history = [], [], []
        # last_action = np.zeros(n_actions) # For smoothing if used

        if render_mode == "rgb_array": # Initial frame for GIF
            try: frame = env.render(); frames.append(frame) if frame is not None else None
            except Exception as e: print(f"Error rendering initial frame: {e}")

        while not (terminated or truncated):
            if current_flat_state is None: break
            action_to_send = agent.choose_action(current_flat_state, evaluate=True) # Updates agent's internal hidden state
            actions_history.append(action_to_send)

            try:
                next_observation_dict, reward, terminated, truncated, info = env.step(action_to_send)
                done = terminated or truncated # Use combined for logic
                next_flat_state = preprocess_observation(next_observation_dict)
            except Exception as e: print(f"Error in env.step: {e}"); traceback.print_exc(); truncated=True; next_flat_state=None; reward=-10.0

            # Store positions, render, update state/score/steps
            # ... (same as before, using current_flat_state and next_flat_state) ...
            actual_pos = info.get('actual_ee_pos'); ref_pos = info.get('reference_ee_pos')
            if actual_pos is not None and ref_pos is not None: actual_positions.append(actual_pos); ref_positions.append(ref_pos)
            if render_mode == "human": time.sleep(delay)
            elif render_mode == "rgb_array":
                try: frame = env.render(); frames.append(frame) if frame is not None else None
                except Exception as e: print(f"Error rendering frame: {e}")
            current_flat_state = next_flat_state; episode_reward += reward; steps += 1
            if done: break
            if steps >= config.MAX_EPISODE_STEPS * 1.1: truncated=True; break

        # --- End of Episode Plotting and MSE calculation ---
        # ... (MSE calculation and Plotting logic for trajectory, error, and actions as before) ...
        total_rewards.append(episode_reward); episode_lengths.append(steps); episode_mse = float('nan'); can_plot = False
        if actual_positions and ref_positions and len(actual_positions) > 1:
            actual_pos_np = np.array(actual_positions); ref_pos_np = np.array(ref_positions); actions_np = np.array(actions_history)
            if actual_pos_np.shape == ref_pos_np.shape and actual_pos_np.ndim == 2 and actual_pos_np.shape[1] == 3 and actions_np.ndim == 2 and actions_np.shape[0] == actual_pos_np.shape[0] and actions_np.shape[1] == n_actions:
                sq_err = np.sum((actual_pos_np - ref_pos_np)**2, axis=1); episode_mse = np.mean(sq_err) if sq_err.size > 0 else 0; can_plot=True
                print(f"Finished Eval Ep {i+1} | Score: {episode_reward:.2f} | Steps: {steps} | MSE: {episode_mse:.6f}")
            else: print(f"Finished Eval Ep {i+1} | ... | MSE: N/A (Shape mismatch)")
        else: print(f"Finished Eval Ep {i+1} | ... | MSE: N/A (No data)")
        all_episode_mse.append(episode_mse)
        if can_plot: # Plotting logic
            try: # Wrap all plotting in try-except
                time_steps = np.arange(len(actual_pos_np)) * (1.0 / config.CONTROL_FREQ)
                # Trajectory Plot
                fig_traj, axs_traj = plt.subplots(3,1,figsize=(10,8),sharex=True); labels=['X','Y','Z'];colors=['r','g','b']
                for dim in range(3): axs_traj[dim].plot(time_steps, ref_pos_np[:,dim],'--',color=colors[dim],label=f'Ref {labels[dim]}'); axs_traj[dim].plot(time_steps, actual_pos_np[:,dim],'-',color=colors[dim],label=f'Actual {labels[dim]}'); axs_traj[dim].set_ylabel(f'{labels[dim]} (m)'); axs_traj[dim].legend(); axs_traj[dim].grid(True)
                axs_traj[2].set_xlabel('Time (s)'); fig_traj.suptitle(f'Trajectory - Ep {i+1} (MSE: {episode_mse:.6f})'); fig_traj.tight_layout(rect=[0,0.03,1,0.97]); fig_traj.savefig(os.path.join(plot_dir,f"eval_ep_{i+1:02d}_traj.png")); plt.close(fig_traj)
                # Error Plot
                err_mag = np.linalg.norm(actual_pos_np - ref_pos_np, axis=1); fig_err, ax_err = plt.subplots(1,1,figsize=(10,4)); ax_err.plot(time_steps, err_mag, label='Error Mag', color='purple'); ax_err.set_xlabel('Time (s)'); ax_err.set_ylabel('Error (m)'); ax_err.set_title(f'Tracking Error - Ep {i+1}'); ax_err.legend(); ax_err.grid(True); ax_err.set_ylim(bottom=0); fig_err.tight_layout(); fig_err.savefig(os.path.join(plot_dir,f"eval_ep_{i+1:02d}_err.png")); plt.close(fig_err)
                # Action Plot
                fig_act, axs_act = plt.subplots(n_actions,1,figsize=(10,2*n_actions),sharex=True);
                if n_actions == 1: axs_act = [axs_act] # Make iterable if only one action
                for adim in range(n_actions): axs_act[adim].plot(time_steps, actions_np[:,adim], label=f'Action[{adim}]'); axs_act[adim].set_ylabel(f'Act[{adim}]'); axs_act[adim].grid(True); axs_act[adim].legend()
                axs_act[-1].set_xlabel('Time (s)'); fig_act.suptitle(f'Actions - Ep {i+1}'); fig_act.tight_layout(rect=[0,0.03,1,0.97]); fig_act.savefig(os.path.join(plot_dir,f"eval_ep_{i+1:02d}_act.png")); plt.close(fig_act)
                print(f"Plots saved for episode {i+1}.")
            except Exception as plot_e: print(f"Error plotting ep {i+1}: {plot_e}"); traceback.print_exc()


    if env: env.close()
    avg_reward = np.mean(total_rewards) if total_rewards else 0; avg_length = np.mean(episode_lengths) if episode_lengths else 0; avg_mse = np.nanmean(all_episode_mse) if all_episode_mse else float('nan')
    print("\n--- Eval Summary ---"); print(f"Episodes: {n_episodes}"); print(f"Avg Score: {avg_reward:.2f}"); print(f"Avg MSE: {avg_mse:.6f}"); print(f"Avg Length: {avg_length:.1f} steps"); print(f"Plots in: {plot_dir}"); print("--------------------")
    if render_mode=="rgb_array" and frames: imageio.mimsave(config.RENDER_FILENAME, frames, duration=max(0.01,config.RENDER_DELAY/1000.0)); print(f"GIF saved: {config.RENDER_FILENAME}")
    return {"avg_score": avg_reward, "avg_mse": avg_mse, "avg_length": avg_length}

if __name__ == "__main__":
    EVAL_RENDER_MODE = "human"; env_name = config.ENV_NAME # Set render mode
    init_env=None; agent=None
    try: init_env = gym.make(env_name, render_mode="rgb_array"); agent = RDPGAgent(env=init_env) # Use RDPGAgent
    except Exception as e: print(f"Fatal Error init: {e}"); traceback.print_exc()
    finally:
        if init_env: init_env.close()
    if agent is None: exit()
    try: agent.load_models(); evaluate_agent(agent, env_name, config.EVAL_EPISODES, EVAL_RENDER_MODE, config.RENDER_DELAY/1000.0)
    except FileNotFoundError: print(f"\nError: Models not found for {config.MODEL_NAME}")
    except Exception as e: print(f"\nError during eval: {e}"); traceback.print_exc()