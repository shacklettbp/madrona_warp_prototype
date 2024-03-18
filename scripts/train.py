import numpy as np
import os
from datetime import datetime
import warp as wp

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

# from . import warp_rls_cartpole_env

import torch

import inspect

import torch
from sim import Simulator
import argparse
from time import time

torch.manual_seed(0)


@wp.kernel
def eval_observations(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    obs[tid, 0] = joint_q[0]
    obs[tid, 1] = joint_q[1]
    obs[tid, 2] = joint_qd[0]
    obs[tid, 3] = joint_qd[1]


@wp.func
def angle_normalize(x: float):
    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    return ((x + wp.pi) % (2.0 * wp.pi)) - wp.pi


@wp.kernel
def eval_rewards(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    rewards: wp.array(dtype=wp.float32),
    dones: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    x = joint_q[env_id * 2 + 0]
    th = joint_q[env_id * 2 + 1]
    thdot = joint_qd[env_id * 2 + 1]
    u = joint_act[env_id * 2]

    c = angle_normalize(th) ** 2.0 + 0.1 * thdot**2.0 + (u * 1e-4) ** 2.0

    rewards[env_id] = -c
    if wp.abs(th) > 0.2 or wp.abs(x) > 0.5:
        dones[env_id] = True
    else:
        dones[env_id] = False


class MadWarpCartpoleCamera(VecEnv):
    """MadWarpCartpoleCamera class for a vectorized cartpole experimental Gym environment."""

    # num_envs: int
    """Number of environments."""
    # num_obs: int
    """Number of observations."""
    # num_privileged_obs: int
    """Number of privileged observations."""
    # num_actions: int
    """Number of actions."""
    # max_episode_length: int
    """Maximum episode length."""
    # privileged_obs_buf: torch.Tensor
    """Buffer for privileged observations."""
    # obs_buf: torch.Tensor
    """Buffer for observations."""
    # rew_buf: torch.Tensor
    """Buffer for rewards."""
    # reset_buf: torch.Tensor
    """Buffer for resets."""
    # episode_length_buf: torch.Tensor  # current episode duration
    """Buffer for current episode lengths."""
    # extras: dict
    """Extra information (metrics).

    Extra information is stored in a dictionary. This includes metrics such as the episode reward, episode length,
    etc. Additional information can be stored in the dictionary such as observations for the critic network, etc.
    """
    # device: torch.device
    """Device to use."""

    """
    Operations.
    """

    def __init__(self, num_envs: int):

        self.num_envs = num_envs
        print("INIT num_envs=", num_envs)

        gpu_id = 0

        use_cpu_sim = False
        self.sim = Simulator(gpu_id, num_envs, use_cpu_sim)

        self.num_obs = 4  # pole, cart position, pole cart velocity
        self.device = "cuda"

        # allocate buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # if self.num_privileged_obs is not None:
        #     self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        # else:
        #     self.privileged_obs_buf = None
        # pass

        self.extras = {}

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return the current observations.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """

        wp.launch(
            eval_observations,
            dim=self.num_envs,
            inputs=[
                self.sim.env_cartpole.state.joint_q,
                self.sim.env_cartpole.state.joint_qd,
            ],
            outputs=[wp.from_torch(self.obs_buf)],
        )

        return self.obs_buf, self.extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """

        self.sim.env_cartpole.reset()
        #  if self._discrete_actions:
        #   force = self.force_mag if action == 1 else -self.force_mag
        # else:
        #   force = action[0]

        # p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force)
        # p.stepSimulation()

        # self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2]
        # theta, theta_dot, x, x_dot = self.state

        # done =  x < -self.x_threshold \
        #             or x > self.x_threshold \
        #             or theta < -self.theta_threshold_radians \
        #             or theta > self.theta_threshold_radians
        # done = bool(done)
        # reward = 1.0
        # #print("state=",self.state)
        # return np.array(self.state), reward, done, {}

        self.sim.env_cartpole.reset()
        return self.get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, rewards, dones and extra information (metrics).
        """
        self.sim.step()
        # A tuple containing the observations, rewards, dones and extra information (metrics).

        wp.launch(
            eval_rewards,
            dim=self.num_envs,
            inputs=[
                self.sim.env_cartpole.state.joint_q,
                self.sim.env_cartpole.state.joint_qd,
                self.sim.env_cartpole.control.joint_act,
            ],
            outputs=[
                wp.from_torch(self.rew_buf),
                wp.from_torch(self.done_buf),
            ],
        )
        self.get_observations()

        return self.obs_buf, self.rew_buf, self.done_buf, self.extras


class BaseConfig:
    def __init__(self) -> None:
        """Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)


class MadWarpRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def train():
    # env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    log_dir = "."

    rl_device = "gpu"

    train_cfg = MadWarpRobotCfgPPO()
    train_cfg_dict = class_to_dict(train_cfg)
    num_envs = 100
    env = MadWarpCartpoleCamera(num_envs)  # warp_rls_cartpole_env.WarpCartpoleEnv()

    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=rl_device)

    ppo_runner.learn(
        num_learning_iterations=ppo_runner.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    # args = get_args()
    train()
