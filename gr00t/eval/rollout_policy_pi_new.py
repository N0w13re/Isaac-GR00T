# import argparse
# from collections import defaultdict
# from dataclasses import dataclass, field
# from functools import partial
# from pathlib import Path
# import time
# from typing import Any
# import uuid

# from gr00t.data.embodiment_tags import EmbodimentTag
# from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
# from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
# from gr00t.policy import BasePolicy
# import gymnasium as gym
# import numpy as np
# from tqdm import tqdm

# from openpi_client import image_tools
# from openpi_client import websocket_client_policy as _websocket_client_policy
# import math
# import collections
# import pandas as pd

# def load_parquet_actions(path):
#     print(f"📼 Loading actions from: {path}")
#     df = pd.read_parquet(path)
#     # 将 list 类型的 column 堆叠成 (Total_Steps, 12) 的 float32 数组
#     return np.stack(df['action'].values).astype(np.float32)

# class ParquetReplayPolicy(BasePolicy):
#     def __init__(self, parquet_path, replan_steps=8):
#         self.actions = load_parquet_actions(parquet_path) # (T, 12)
#         self.cursor = 0
#         self.replan_steps = replan_steps
#         self.total = len(self.actions)

#     def reset(self):
#         self.cursor = 0
    
#     def check_observation(self, observation):
#         pass

#     def check_action(self, action):
#         pass

#     def _get_action(self, *args, **kwargs):
#         pass

#     def get_action(self, observations):
#         # 1. 切片获取当前 chunk
#         chunk = self.actions[self.cursor : self.cursor + self.replan_steps]
        
#         # 2. 如果数据读完或不够，用最后一帧补齐 (防止报错)
#         if len(chunk) < self.replan_steps:
#             pad_len = self.replan_steps - len(chunk)
#             last_frame = self.actions[-1] if self.total > 0 else np.zeros(12)
#             pad = np.tile(last_frame, (pad_len, 1))
#             chunk = np.vstack([chunk, pad]) if len(chunk) > 0 else pad

#         self.cursor += self.replan_steps

#         # 3. 拆分并增加 batch 维度 (1, T, D) 适配单环境
#         return {
#             "action.base_motion": chunk[:, 0:4][None],           # (1, 8, 4)
#             "action.control_mode": chunk[:, 4:5][None],          # (1, 8, 1)
#             "action.end_effector_position": chunk[:, 5:8][None], # (1, 8, 3)
#             "action.end_effector_rotation": chunk[:, 8:11][None],# (1, 8, 3)
#             "action.gripper_close": chunk[:, 11:12][None],       # (1, 8, 1)
#         }, {}


# def print_green(text):
#     print(f"\033[92m{text}\033[0m")

# def print_blue(text):
#     print(f"\033[94m{text}\033[0m")

# class OpenPIPolicyAdapter(BasePolicy):
#     def __init__(self, host, port, resize_size=224, replan_steps=8):
#         self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
#         self.resize_size = resize_size
#         self.replan_steps = replan_steps
#         self.action_queues = defaultdict(lambda: collections.deque())

#     def reset(self):
#         self.action_queues.clear()

#     def check_observation(self, observation):
#         pass

#     def check_action(self, action):
#         pass

#     def _get_action(self, *args, **kwargs):
#         pass

#     def get_action(self, observations):
#         if "state.base_position" not in observations:
#              print("Available keys:", observations.keys())
#              raise ValueError("Missing 'state.base_position' in observations")
             
#         n_envs = len(observations["state.base_position"])

#         for k, v in observations.items():
#             print_green(k)
#             print_green(v.shape if hasattr(v, 'shape') else type(v))

#         batch_elements = []
        
#         for i in range(n_envs):
#             # 取出第 i 个环境、最新一帧 (-1) 的图像
#             img_wrist_raw = observations["video.res256_image_wrist_0"][i, -1]
#             img_left_raw  = observations["video.res256_image_side_0"][i, -1]
#             img_right_raw = observations["video.res256_image_side_1"][i, -1]

#             img_wrist = self._process_image(img_wrist_raw)      # 不需要翻转！
#             img_left  = self._process_image(img_left_raw)
#             img_right = self._process_image(img_right_raw)

#             state_parts = [
#                 observations["state.base_position"][i, -1],                # (3,)
#                 observations["state.base_rotation"][i, -1],                # (4,)
#                 observations["state.end_effector_position_absolute"][i, -1], # (3,)
#                 observations["state.end_effector_position_relative"][i, -1], # (3,)
#                 observations["state.end_effector_rotation_absolute"][i, -1], # (4,)
#                 observations["state.end_effector_rotation_relative"][i, -1], # (4,)
#                 observations["state.gripper_qpos"][i, -1],                 # (2,)
#                 observations["state.gripper_qvel"][i, -1],                 # (2,)
#                 observations["state.joint_position"][i, -1],               # (7,)
#                 observations["state.joint_position_cos"][i, -1],                # (7,)
#                 observations["state.joint_position_sin"][i, -1],                # (7,)
#                 observations["state.joint_velocity"][i, -1],               # (7,)
#             ]
            
#             state_vec = np.concatenate(state_parts, axis=-1)
#             task_description = observations["annotation.human.action.task_description"][i]

#             element = {
#                 "cam_high": img_wrist,
#                 "cam_left_wrist": img_left,
#                 "cam_right_wrist": img_right,
#                 "state": state_vec,
#                 "prompt": task_description,
#             }
            
#             batch_elements.append(element)

#         # 推理
#         output = self.client.infer(batch_elements)
#         action_chunk = output["actions"]

#         print_blue("action_chunk")
#         print_blue(action_chunk.shape)

#         steps_to_take = min(len(action_chunk), self.replan_steps)
        
#         final_actions = dict()
#         final_actions["action.base_motion"] = action_chunk[:, :steps_to_take, 0:4]
#         final_actions["action.control_mode"] = action_chunk[:, :steps_to_take, 4:5]
#         final_actions["action.end_effector_position"] = action_chunk[:, :steps_to_take, 5:8]
#         final_actions["action.end_effector_rotation"] = action_chunk[:, :steps_to_take, 8:11]
#         final_actions["action.gripper_close"] = action_chunk[:, :steps_to_take, 11:12]
        
#         # print_green(final_actions)
#         return final_actions, {}

#     def _process_image(self, img_array):
#         # (H, W, C) -> (1, H, W, C) -> Resize -> (C, H, W)
#         img_batched = img_array[np.newaxis, ...]
#         resized = image_tools.resize_with_pad(img_batched, self.resize_size, self.resize_size)
#         return image_tools.convert_to_uint8(resized[0]).transpose(2, 0, 1)


# @dataclass
# class VideoConfig:
#     """Configuration for video recording settings.

#     Attributes:
#         video_dir: Directory to save videos (if None, no videos are saved)
#         steps_per_render: Number of steps between each call to env.render() while recording
#             during rollout
#         fps: Frames per second for the output video
#         codec: Video codec to use for compression
#         input_pix_fmt: Input pixel format
#         crf: Constant Rate Factor for video compression (lower = better quality)
#         thread_type: Threading strategy for video encoding
#         thread_count: Number of threads to use for encoding
#     """

#     video_dir: str | None = None
#     steps_per_render: int = 2
#     max_episode_steps: int = 720
#     fps: int = 20
#     codec: str = "h264"
#     input_pix_fmt: str = "rgb24"
#     crf: int = 22
#     thread_type: str = "FRAME"
#     thread_count: int = 1
#     overlay_text: bool = True
#     n_action_steps: int = 8


# @dataclass
# class MultiStepConfig:
#     """Configuration for multi-step environment settings.

#     Attributes:
#         video_delta_indices: Indices of video observations to stack
#         state_delta_indices: Indices of state observations to stack
#         n_action_steps: Number of action steps to execute
#         max_episode_steps: Maximum number of steps per episode
#     """

#     video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
#     state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
#     n_action_steps: int = 16
#     max_episode_steps: int = 720
#     terminate_on_success: bool = False


# @dataclass
# class WrapperConfigs:
#     """Container for various environment wrapper configurations.

#     Attributes:
#         video: Configuration for video recording
#         multistep: Configuration for multi-step processing
#     """

#     video: VideoConfig = field(default_factory=VideoConfig)
#     multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


# def get_robocasa_env_fn(
#     env_name: str,
# ):
#     def env_fn():
#         import os

#         import robocasa  # noqa: F401
#         from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
#         import robosuite  # noqa: F401

#         os.environ["MUJOCO_GL"] = "egl"
#         return gym.make(env_name, enable_render=True)

#     return env_fn


# def get_groot_locomanip_env_fn(
#     env_name: str,
# ):
#     def env_fn():
#         from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
#         from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
#         from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
#         import robocasa  # noqa: F401

#         gym_env = gym.make(
#             env_name,
#             onscreen=False,
#             offscreen=True,
#             enable_waist=True,
#             randomize_cameras=False,
#             camera_names=[
#                 "robot0_oak_egoview",
#                 "robot0_rs_tppview",
#             ],
#         )
#         wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
#         gym_env = WholeBodyControlWrapper(gym_env, wbc_config)
#         return gym_env

#     return env_fn


# def get_simpler_env_fn(
#     env_name: str,
# ):
#     def env_fn():
#         from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

#         register_simpler_envs()
#         return gym.make(env_name)

#     return env_fn


# def get_libero_env_fn(
#     env_name: str,
# ):
#     def env_fn():
#         from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

#         register_libero_envs()
#         return gym.make(env_name)

#     return env_fn


# def get_behavior_env_fn(
#     env_name: str,
#     env_idx: int,
#     total_n_envs: int,
# ):
#     def env_fn():
#         from gr00t.eval.sim.BEHAVIOR.behavior_env import register_behavior_envs

#         register_behavior_envs()
#         return gym.make(env_name, env_idx=env_idx, total_n_envs=total_n_envs)

#     return env_fn


# def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
#     """Create Ray environment factory function without wrappers."""

#     env_embodiment = get_embodiment_tag_from_env_name(env_name)

#     if env_embodiment in (
#         EmbodimentTag.GR1,
#         EmbodimentTag.ROBOCASA_PANDA_OMRON,
#     ):
#         env_fn = get_robocasa_env_fn(env_name)

#     elif env_embodiment in (EmbodimentTag.UNITREE_G1,):
#         env_fn = get_groot_locomanip_env_fn(env_name)

#     elif env_embodiment in (EmbodimentTag.OXE_GOOGLE, EmbodimentTag.OXE_WIDOWX):
#         env_fn = get_simpler_env_fn(env_name)

#     elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
#         env_fn = get_libero_env_fn(env_name)

#     elif env_embodiment in (EmbodimentTag.BEHAVIOR_R1_PRO,):
#         env_fn = get_behavior_env_fn(env_name, env_idx, total_n_envs)
#     else:
#         raise ValueError(f"Invalid environment name: {env_name}")

#     return env_fn()


# def create_eval_env(
#     env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
# ) -> gym.Env:
#     """Create a single evaluation environment with wrappers.

#     Args:
#         env_name: Name of the gymnasium environment to use
#         idx: Environment index (used to determine video recording)
#         wrapper_configs: Configuration for environment wrappers
#     Returns:
#         Wrapped gymnasium environment
#     """

#     env = get_gym_env(env_name, env_idx, total_n_envs)
#     if wrapper_configs.video.video_dir is not None:
#         from gr00t.eval.sim.wrapper.video_recording_wrapper import (
#             VideoRecorder,
#             VideoRecordingWrapper,
#         )

#         video_recorder = VideoRecorder.create_h264(
#             fps=wrapper_configs.video.fps,
#             codec=wrapper_configs.video.codec,
#             input_pix_fmt=wrapper_configs.video.input_pix_fmt,
#             crf=wrapper_configs.video.crf,
#             thread_type=wrapper_configs.video.thread_type,
#             thread_count=wrapper_configs.video.thread_count,
#         )
#         env = VideoRecordingWrapper(
#             env,
#             video_recorder,
#             video_dir=Path(wrapper_configs.video.video_dir),
#             steps_per_render=wrapper_configs.video.steps_per_render,
#             max_episode_steps=wrapper_configs.video.max_episode_steps,
#             overlay_text=wrapper_configs.video.overlay_text,
#         )

#     env = MultiStepWrapper(
#         env,
#         video_delta_indices=wrapper_configs.multistep.video_delta_indices,
#         state_delta_indices=wrapper_configs.multistep.state_delta_indices,
#         n_action_steps=wrapper_configs.multistep.n_action_steps,
#         max_episode_steps=wrapper_configs.multistep.max_episode_steps,
#         terminate_on_success=wrapper_configs.multistep.terminate_on_success,
#     )
#     return env


# def run_rollout_gymnasium_policy(
#     env_name: str,
#     policy: BasePolicy,
#     wrapper_configs: WrapperConfigs,
#     n_episodes: int = 10,
#     n_envs: int = 1,
# ) -> Any:
#     """Run policy rollouts in parallel environments.

#     Args:
#         env_name: Name of the gymnasium environment to use
#         policy_fn: Function that creates a policy instance
#         n_episodes: Number of episodes to run
#         n_envs: Number of parallel environments
#         wrapper_configs: Configuration for environment wrappers
#         ray_env: Whether to use ray gym env to create each env.
#     Returns:
#         Collection results from running the episodes
#     """
#     start_time = time.time()
#     n_episodes = max(n_episodes, n_envs)
#     print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

#     env_fns = [
#         partial(
#             create_eval_env,
#             env_idx=idx,
#             env_name=env_name,
#             total_n_envs=n_envs,
#             wrapper_configs=wrapper_configs,
#         )
#         for idx in range(n_envs)
#     ]

#     if n_envs == 1:
#         env = gym.vector.SyncVectorEnv(env_fns)
#     else:
#         env = gym.vector.AsyncVectorEnv(
#             env_fns,
#             shared_memory=False,
#             context="spawn",
#         )

#     # Storage for results
#     episode_lengths = []
#     current_rewards = [0] * n_envs
#     current_lengths = [0] * n_envs
#     completed_episodes = 0
#     current_successes = [False] * n_envs
#     episode_successes = []
#     episode_infos = defaultdict(list)

#     # Initial reset
#     observations, _ = env.reset()
    
#     policy.reset()
#     i = 0

#     pbar = tqdm(total=n_episodes, desc="Episodes")
#     while completed_episodes < n_episodes:
#         actions, _ = policy.get_action(observations)
#         # print_green("base_motion")
#         # print_blue(actions["action.base_motion"])
#         # print_green("control_mode")
#         # print_blue(actions["action.control_mode"])
#         # print_green("end_effector_position")
#         # print_blue(actions["action.end_effector_position"])
#         # print_green("end_effector_rotation")
#         # print_blue(actions["action.end_effector_rotation"])
#         # print_green("gripper_close")
#         # print_blue(actions["action.gripper_close"])
#         # exit()
#         next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
#         # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
#         # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
#         i += 1
#         # Update episode tracking
#         for env_idx in range(n_envs):
#             if "success" in env_infos:
#                 env_success = env_infos["success"][env_idx]
#                 if isinstance(env_success, list):
#                     env_success = np.any(env_success)
#                 elif isinstance(env_success, np.ndarray):
#                     env_success = np.any(env_success)
#                 elif isinstance(env_success, bool):
#                     env_success = env_success
#                 elif isinstance(env_success, int):
#                     env_success = bool(env_success)
#                 else:
#                     raise ValueError(f"Unknown success dtype: {type(env_success)}")
#                 current_successes[env_idx] |= bool(env_success)
#             else:
#                 current_successes[env_idx] = False

#             if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
#                 env_success = env_infos["final_info"][env_idx]["success"]
#                 if isinstance(env_success, list):
#                     env_success = any(env_success)
#                 elif isinstance(env_success, np.ndarray):
#                     env_success = np.any(env_success)
#                 elif isinstance(env_success, bool):
#                     env_success = env_success
#                 elif isinstance(env_success, int):
#                     env_success = bool(env_success)
#                 else:
#                     raise ValueError(f"Unknown success dtype: {type(env_success)}")
#                 current_successes[env_idx] |= bool(env_success)
#             current_rewards[env_idx] += rewards[env_idx]
#             current_lengths[env_idx] += 1

#             # If episode ended, store results
#             if terminations[env_idx] or truncations[env_idx]:
#                 if "final_info" in env_infos:
#                     current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
#                 if "task_progress" in env_infos:
#                     episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
#                 if "q_score" in env_infos:
#                     episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
#                 if "valid" in env_infos:
#                     episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
#                 # Accumulate results
#                 episode_lengths.append(current_lengths[env_idx])
#                 episode_successes.append(current_successes[env_idx])
#                 # Reset trackers for this environment.
#                 current_successes[env_idx] = False
#                 # only update completed_episodes if valid
#                 if "valid" in episode_infos:
#                     if episode_infos["valid"][-1]:
#                         completed_episodes += 1
#                         pbar.update(1)
#                 else:
#                     # envs don't return valid
#                     completed_episodes += 1
#                     pbar.update(1)
#                 current_rewards[env_idx] = 0
#                 current_lengths[env_idx] = 0
#         observations = next_obs
#     pbar.close()

#     env.reset()
#     env.close()
#     print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

#     assert len(episode_successes) >= n_episodes, (
#         f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
#     )

#     episode_infos = dict(episode_infos)  # Convert defaultdict to dict
#     for key, value in episode_infos.items():
#         assert len(value) == len(episode_successes), (
#             f"Length of {key} is not equal to the number of episodes"
#         )

#     # process valid results
#     if "valid" in episode_infos:
#         valids = episode_infos["valid"]
#         valid_idxs = np.where(valids)[0]
#         episode_successes = [episode_successes[i] for i in valid_idxs]
#         episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

#     return env_name, episode_successes, episode_infos


# def create_gr00t_sim_policy(
#     model_path: str,
#     embodiment_tag: EmbodimentTag,
#     policy_client_host: str = "",
#     policy_client_port: int | None = None,
#     n_action_steps: int | None = None
# ) -> BasePolicy:
#     from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

#     # replay_path = "/pi/episode_000000.parquet"
#     # if replay_path:
#     #     return ParquetReplayPolicy(replay_path, replan_steps=n_action_steps)

#     if policy_client_host and policy_client_port:
#         print(f"🔌 Connecting to OpenPI Server at {policy_client_host}:{policy_client_port}")
#         return OpenPIPolicyAdapter(
#             host=policy_client_host, 
#             port=policy_client_port,
#             replan_steps=n_action_steps,
#         )
#     else:
#         policy = Gr00tSimPolicyWrapper(
#             Gr00tPolicy(
#                 embodiment_tag=embodiment_tag,
#                 model_path=model_path,
#                 device=0,
#             )
#         )
#     return policy


# def run_gr00t_sim_policy(
#     env_name: str,
#     n_episodes: int,
#     max_episode_steps: int,
#     model_path: str = "",
#     policy_client_host: str = "",
#     policy_client_port: int | None = None,
#     n_envs: int = 8,
#     n_action_steps: int = 8,
# ):
#     embodiment_tag = get_embodiment_tag_from_env_name(env_name)

#     if model_path:
#         video_dir = (
#             f"/tmp/sim_eval_videos_{model_path.split('/')[-3]}_ac{n_action_steps}_{uuid.uuid4()}"
#         )
#     else:
#         video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
#     if env_name.startswith("sim_behavior_r1_pro"):
#         # BEHAVIOR sim will crash if decord is imported in video_utils.py
#         video_dir = None
#     wrapper_configs = WrapperConfigs(
#         video=VideoConfig(
#             video_dir=video_dir,
#             max_episode_steps=max_episode_steps,
#         ),
#         multistep=MultiStepConfig(
#             n_action_steps=n_action_steps,
#             max_episode_steps=max_episode_steps,
#             terminate_on_success=True,
#         ),
#     )

#     policy = create_gr00t_sim_policy(
#         model_path, embodiment_tag, policy_client_host, policy_client_port, n_action_steps
#     )

#     results = run_rollout_gymnasium_policy(
#         env_name=env_name,
#         policy=policy,
#         wrapper_configs=wrapper_configs,
#         n_episodes=n_episodes,
#         n_envs=n_envs,
#     )
#     print("Video saved to: ", wrapper_configs.video.video_dir)
#     return results


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_episode_steps", type=int, default=504)
#     parser.add_argument("--n_episodes", type=int, default=50)
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="",
#     )
#     parser.add_argument("--policy_client_host", type=str, default="")
#     parser.add_argument("--policy_client_port", type=int, default=None)
#     parser.add_argument(
#         "--env_name",
#         type=str,
#         default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
#     )
#     parser.add_argument("--n_envs", type=int, default=8)
#     parser.add_argument("--n_action_steps", type=int, default=8)

#     args = parser.parse_args()

#     # validate policy configuration
#     assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
#         not args.model_path and args.policy_client_host and args.policy_client_port is not None
#     ), (
#         "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
#         "If all 3 arguments are provided, explicitly choose one:\n"
#         '  - To use policy client: set --policy_client_host and --policy_client_port, and set --model_path ""\n'
#         '  - To use model path: set --model_path, and set --policy_client_host "" (and leave --policy_client_port unset)'
#     )

#     results = run_gr00t_sim_policy(
#         env_name=args.env_name,
#         n_episodes=args.n_episodes,
#         max_episode_steps=args.max_episode_steps,
#         model_path=args.model_path,
#         policy_client_host=args.policy_client_host,
#         policy_client_port=args.policy_client_port,
#         n_envs=args.n_envs,
#         n_action_steps=args.n_action_steps,
#     )
#     print("results: ", results)
#     print("success rate: ", np.mean(results[1]))
    
#     import os
#     os.makedirs("/pi/robocasa_eval_results", exist_ok=True)
#     file_name = "/pi/robocasa_eval_results/0210.txt"
#     with open (file_name, "a") as f:
#         f.write(
#             f"task_name: {results[0]}\t"+
#             f"n_episodes: {args.n_episodes}\t"+
#             f"n_envs: {args.n_envs}\t"+
#             f"success_rate: {np.mean(results[1])}\n"
#         )
