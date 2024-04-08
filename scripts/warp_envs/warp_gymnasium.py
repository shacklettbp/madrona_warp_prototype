import gymnasium as gym

import numpy as np
import warp as wp

from .environment import Environment, RenderMode


def register_gym_env(env_name, Env, traj_length=500, env_args={}, registration_kwargs={}):

    class GymEnv(gym.Env):
        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        def __init__(self, render_mode=None):
            env_render_mode = RenderMode.NONE
            if render_mode == "human" or render_mode == "rgb_array":
                env_render_mode = RenderMode.OPENGL
            env_args["render_mode"] = env_render_mode
            env_args["num_envs"] = 1
            self.env: Environment = Env(**env_args)
            self.traj_length = traj_length

            self.observation_space = gym.spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(self.env.observation_dim,),
            )

            ctrl_limits = np.array(self.env.control_limits)
            print("ctrl_limits=",ctrl_limits)
            self.action_space = gym.spaces.Box(low=ctrl_limits[:, 0], high=ctrl_limits[:, 1])

            assert render_mode is None or render_mode in self.metadata["render_modes"]
            self.render_mode = render_mode

            if self.env.use_graph_capture:
                with wp.ScopedCapture() as capture:
                    self.env.update()
                self.graph = capture.graph
            else:
                self.graph = None

            self._termination_buffer = wp.zeros(1, dtype=wp.bool)
            self._termination_buffer_cpu = wp.zeros(1, dtype=wp.bool, device="cpu", pinned=True)
            self._observation_buffer = wp.zeros((1, self.env.observation_dim), dtype=wp.float32)
            self._observation_buffer_cpu = wp.zeros(
                (1, self.env.observation_dim), dtype=wp.float32, device="cpu", pinned=True
            )
            self._cost_buffer = wp.zeros(1, dtype=wp.float32)
            self._cost_buffer_cpu = wp.zeros(1, dtype=wp.float32, device="cpu", pinned=True)

            self._step_count = 0

        def _get_obs(self):
            self.env.compute_observations(self.env.state, self.env.control, self._observation_buffer, 0, 0)
            wp.copy(self._observation_buffer_cpu, self._observation_buffer)
            observation = self._observation_buffer_cpu.numpy()[0]
            return observation

        def _get_info(self):
            return {}
            # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

        def reset(self, seed=None, options=None):
            # We need the following line to seed self.np_random
            super().reset(seed=seed)

            self.env.reset()
            self._step_count = 0

            observation = self._get_obs()
            info = {}

            if self.render_mode == "human":
                self._render_frame()

            return observation, info

        @property
        def has_terminated(self):
            wp.copy(self._termination_buffer_cpu, self._termination_buffer)
            return self._termination_buffer_cpu.numpy()[0]

        def step(self, action):
            self.env.control_input.assign(action * self.env.control_gains)
            if self.env.use_graph_capture:
                wp.capture_launch(self.graph)
            else:
                self.env.update()

            # reward is single-step, not cumulative
            self._cost_buffer.zero_()
            self.env.compute_cost_termination(
                self.env.state,
                self.env.control,
                self._step_count,
                self.traj_length,
                self._cost_buffer,
                self._termination_buffer,
            )
            terminated = self._step_count >= self.traj_length or self.has_terminated
            wp.copy(self._cost_buffer_cpu, self._cost_buffer)
            reward = 10.0 - self._cost_buffer_cpu.numpy()[0]
            observation = self._get_obs()

            if self.render_mode == "human":
                self._render_frame()

            self._step_count += 1

            info = {}
            return observation, reward, terminated, False, info

        def render(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()

        def _render_frame(self):
            self.env.render()

            if self.render_mode == "rgb_array":
                success = self.renderer.get_pixels(self._pixel_buffer, split_up_tiles=True, mode="rgb")
                assert success

                wp.copy(self._pixel_buffer_cpu, self._pixel_buffer)

                return self._pixel_buffer_cpu[0]

        def close(self):
            self.env.renderer.close()

    gym.envs.registration.register(id=env_name, entry_point=GymEnv, **registration_kwargs)

    return GymEnv


from .env_cartpole import CartpoleEnvironment

CartpoleEnv = register_gym_env("WarpSimCartPole-v1", CartpoleEnvironment)
