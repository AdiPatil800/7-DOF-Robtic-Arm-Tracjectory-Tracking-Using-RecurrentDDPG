# custom_panda_env.py
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
import traceback
import sys

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Warning: pybullet package not found. Debug visualizations will be disabled.")

# Attempt to import panda_gym components safely
try:
    from panda_gym.envs.core import RobotTaskEnv, Task
    from panda_gym.envs.robots.panda import Panda
    from panda_gym.pybullet import PyBullet
except ImportError as e:
    print("="*80); print("Error importing panda_gym."); print(f"ImportError: {e}"); print("="*80); raise

# --- Define trajectory parameters ---
TRAJ_CENTER = np.array([0.3, 0.0, 0.2])
TRAJ_RADIUS = 0.15
TRAJ_FREQ = 0.1 # Hz (10 seconds for one full circle)
TRAJ_Z_AMP = 0.05
TRAJ_DURATION = 10.0 # seconds
SIM_FREQ = 240 # Hz (Pybullet simulation frequency)
CONTROL_FREQ = 20 # Hz (Agent action frequency)
MAX_EPISODE_STEPS = int(TRAJ_DURATION * CONTROL_FREQ)

# Trajectory function
def get_sinusoid_trajectory_point(t, cx=TRAJ_CENTER[0], cy=TRAJ_CENTER[1], cz=TRAJ_CENTER[2], r=TRAJ_RADIUS, f=TRAJ_FREQ, az=TRAJ_Z_AMP):
    angle = 2 * np.pi * f * t
    target_x = cx + r * np.cos(angle); target_y = cy + r * np.sin(angle); target_z = cz + az * np.sin(2 * angle)
    return np.array([target_x, target_y, target_z])

# --- Define the Task Class FIRST ---
class SinusoidTrackTask(Task):
    def __init__(self, sim, robot):
        super().__init__(sim)
        self.robot = robot
        self.sim_time = 0.0
        self.goal_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.goal = self._get_goal()

    def _get_goal(self):
        calc_time = min(self.sim_time, TRAJ_DURATION)
        return get_sinusoid_trajectory_point(calc_time)

    def get_obs(self): return self.goal.astype(np.float32)
    def get_achieved_goal(self):
        if hasattr(self.robot, 'get_ee_position'): return np.array(self.robot.get_ee_position()).astype(np.float32)
        else: print("Error: Task's robot object missing get_ee_position."); return np.zeros(3, dtype=np.float32)
    def compute_reward(self, achieved_goal, desired_goal, info): return -np.sum((np.array(achieved_goal) - np.array(desired_goal))**2)
    def is_success(self, achieved_goal, desired_goal, info=None): return False
    def reset(self): self.sim_time = 0.0; self.goal = self._get_goal(); return self.get_obs()


# --- Define the Environment Class SECOND ---
class PandaSinusoidTrackEnv(RobotTaskEnv):
    def __init__(self, render_mode=None, control_type="ee"):
        try:
            self.sim_time = 0.0
            self.current_step = 0
            self._control_type = control_type
            self.sim_steps_per_control = SIM_FREQ // CONTROL_FREQ
            self.render_mode = render_mode
            self.target_marker_id = None
            self.trajectory_line_ids = []

            sim = PyBullet(render_mode=self.render_mode, n_substeps=self.sim_steps_per_control)
            self.sim = sim
            robot = Panda(sim=self.sim, block_gripper=True, control_type=self._control_type)
            self.robot = robot
            task = SinusoidTrackTask(sim=self.sim, robot=self.robot)
            self.task = task

            # --- Define observation_space BEFORE super().__init__() ---
            # Based on debug output, self.robot.get_obs() returns a (6,) array for "ee" control
            # This typically means 3D EE position + 3D EE orientation (Euler/RPY)
            # This (6,) vector will be the "observation" part of our Dict space.
            robot_kinematic_obs_shape = (6,) # From debug output
            robot_state_observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=robot_kinematic_obs_shape, dtype=np.float64 # Match robot.get_obs() dtype
            )

            if not hasattr(self.task, 'goal_space'):
                raise AttributeError("Task object missing goal_space.")

            self.observation_space = gym.spaces.Dict(
                dict(
                    observation=robot_state_observation_space,  # The (6,) vector from robot.get_obs()
                    achieved_goal=self.task.goal_space,       # Shape (3,) - EE position
                    desired_goal=self.task.goal_space        # Shape (3,) - Target position
                )
            )
            print(f"DEBUG: PandaSinusoidTrackEnv observation_space defined: {self.observation_space}")

            # Action space is set by the robot, will be available after super().__init__
            # or can be taken from self.robot directly if needed earlier.
            # For now, we'll let super().__init__ handle setting self.action_space.

            # Initialize RobotTaskEnv base class
            super().__init__(robot=self.robot, task=self.task)

            # Store frequencies AFTER super init
            self.control_frequency = CONTROL_FREQ

            # Ensure action_space is set (usually by super().__init__)
            if not hasattr(self, 'action_space'):
                if hasattr(self.robot, 'action_space'):
                    self.action_space = self.robot.action_space
                    print("DEBUG: Manually set action_space from robot after super().__init__.")
                else:
                    raise AttributeError("Environment has no action_space defined after initialization.")

        except Exception as e:
             print("\n*** Error during PandaSinusoidTrackEnv __init__ ***")
             traceback.print_exc()
             if hasattr(self, 'sim') and self.sim is not None:
                  try: self.sim.close()
                  except Exception: pass
             raise e

    def _get_target_position(self):
        calc_time = min(self.sim_time, TRAJ_DURATION)
        return get_sinusoid_trajectory_point(calc_time)

    def _get_obs(self):
        """Return observation dictionary.
           'observation' key contains the robot's raw (6D) state.
           'achieved_goal' is current EE position.
           'desired_goal' is current target sinusoid position.
        """
        try:
            robot_obs_vector = self.robot.get_obs() # This is the (6,) numpy array
            if not isinstance(robot_obs_vector, np.ndarray) or robot_obs_vector.shape != self.observation_space["observation"].shape:
                print(f"Warning: Mismatch in _get_obs! Robot obs shape: {robot_obs_vector.shape}, Expected: {self.observation_space['observation'].shape}")
                # Fallback to a zero array of the correct shape
                robot_obs_vector = np.zeros(self.observation_space["observation"].shape, dtype=np.float32)


            achieved_goal = self.task.get_achieved_goal() # Current EE pos from Task method
            desired_goal = self._get_target_position() # Current target pos based on env time
            self.task.goal = desired_goal # Update task's internal goal state for consistency

            obs_dict = {
                "observation": robot_obs_vector.astype(np.float32), # robot's 6D state
                "achieved_goal": achieved_goal.astype(np.float32),
                "desired_goal": desired_goal.astype(np.float32)
            }
            return obs_dict
        except Exception as e:
             print(f"Error in _get_obs: {e}"); traceback.print_exc()
             dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
             return dummy_obs

    def _get_info(self):
        try:
            achieved_pos = self.task.get_achieved_goal(); target_pos = self._get_target_position()
            distance_error = np.linalg.norm(achieved_pos - target_pos) if achieved_pos is not None and target_pos is not None else float('inf')
            is_success = self.task.is_success(achieved_pos, target_pos)
            return {"actual_ee_pos": achieved_pos if achieved_pos is not None else np.zeros(3),
                    "reference_ee_pos": target_pos if target_pos is not None else np.zeros(3),
                    "distance_error": distance_error, "sim_time": self.sim_time, "is_success": is_success }
        except Exception as e: print(f"Error in _get_info: {e}"); traceback.print_exc(); return { "actual_ee_pos": np.zeros(3), "reference_ee_pos": np.zeros(3), "distance_error": float('inf'), "sim_time": self.sim_time, "is_success": False }

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.sim_time = 0.0
            self.current_step = 0
            self.task.reset()
            observation = self._get_obs()
            info = self._get_info()
            if self.render_mode == "human" and PYBULLET_AVAILABLE:
                self._remove_visualizations(); self._draw_trajectory_line(); self._create_target_marker()
                initial_target_pos = info.get("reference_ee_pos", self._get_target_position())
                self._update_target_marker(initial_target_pos)
            return observation, info
        except Exception as e:
             print("\n*** Error during reset ***"); traceback.print_exc(); print("*"*47+"\n")
             if hasattr(self, 'observation_space') and isinstance(self.observation_space, gym.spaces.Dict):
                 dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
                 return dummy_obs, {}
             else: print("Cannot return dummy obs from reset as observation_space not defined/Dict."); raise e

    def step(self, action):
        terminated = False; truncated = False
        try:
            self.robot.set_action(action); self.sim.step()
            self.sim_time += 1.0 / self.control_frequency; self.current_step += 1
            observation = self._get_obs(); info = self._get_info()
            if self.render_mode == "human" and PYBULLET_AVAILABLE:
                self._update_target_marker(info["reference_ee_pos"])
            achieved_goal = info["actual_ee_pos"]; desired_goal = info["reference_ee_pos"]
            reward = self.task.compute_reward(achieved_goal, desired_goal, info)
            terminated = info["is_success"]; truncated = self.current_step >= MAX_EPISODE_STEPS
            # self.render() # Not needed for human mode if sim steps render
            return observation, reward, terminated, truncated, info
        except Exception as e:
             print(f"\n*** Error during step {self.current_step} ***"); traceback.print_exc(); print("*"*47+"\n")
             if hasattr(self, 'observation_space') and isinstance(self.observation_space, gym.spaces.Dict):
                 dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
                 return dummy_obs, -100.0, False, True, {"is_success": False}
             else: print("Cannot return dummy obs from step as observation_space not defined/Dict."); raise e


    def render(self):
        if hasattr(self, 'sim') and self.sim is not None: return self.sim.render()
        return None
    def close(self):
        if self.render_mode == "human" and PYBULLET_AVAILABLE: self._remove_visualizations()
        if hasattr(self, 'sim') and self.sim is not None: self.sim.close()

    def _remove_visualizations(self):
        if not PYBULLET_AVAILABLE: return
        for line_id in self.trajectory_line_ids:
            try: p.removeUserDebugItem(line_id)
            except: pass
        self.trajectory_line_ids = []
        if self.target_marker_id is not None:
            try: p.removeBody(self.target_marker_id)
            except: pass
            self.target_marker_id = None
    def _draw_trajectory_line(self, num_points=100, color=[1, 0.8, 0], width=2):
        if not PYBULLET_AVAILABLE: return
        points = [get_sinusoid_trajectory_point(t) for t in np.linspace(0, TRAJ_DURATION, num_points)]
        for i in range(len(points) - 1):
            try: line_id = p.addUserDebugLine( points[i], points[i+1], lineColorRGB=color, lineWidth=width); self.trajectory_line_ids.append(line_id)
            except Exception as e: print(f"Error adding debug line segment {i}: {e}"); break
    def _create_target_marker(self, radius=0.02, color=[1, 0.8, 0, 0.7]):
        if not PYBULLET_AVAILABLE: return
        try:
            if self.target_marker_id is not None:
                 try: p.removeBody(self.target_marker_id)
                 except: pass
            visual_shape_id = p.createVisualShape( shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            self.target_marker_id = p.createMultiBody( baseVisualShapeIndex=visual_shape_id, baseCollisionShapeIndex=-1, basePosition=[0,0,-1], useMaximalCoordinates=False)
        except Exception as e: print(f"Error creating target marker: {e}"); self.target_marker_id = None
    def _update_target_marker(self, target_position):
        if self.target_marker_id is not None and PYBULLET_AVAILABLE:
            try: p.resetBasePositionAndOrientation( self.target_marker_id, posObj=target_position, ornObj=[0,0,0,1])
            except Exception as e: pass

# --- Registration ---
try:
    gym.register( id='PandaSinusoidTrack-v0', entry_point='custom_panda_env:PandaSinusoidTrackEnv', max_episode_steps=MAX_EPISODE_STEPS )
    print("Custom Environment 'PandaSinusoidTrack-v0' (operational init) registered.")
except Exception as e: print(f"Error registering environment: {e}")