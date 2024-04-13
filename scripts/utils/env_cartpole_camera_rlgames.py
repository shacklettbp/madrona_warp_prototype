from madrona_warp_proto_sim import SimManager, madrona
import math
from dataclasses import dataclass

import numpy as np
import nvtx
import warp as wp
import warp.sim
import warp.sim.render
from warp.torch import to_torch

#from warp_envs.assets import get_asset_path
#from warp_envs.envs.base import WarpEnv, WarpEnvConfig

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ctypes
import math
import random
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from torchvision.utils import save_image

import numpy as np
import nvtx
import torch
import warp as wp
from gym import spaces

import warp as wp
import math

# initialize the Warp runtime
# this function must be called before any other Warp API call.
wp.config.mode = "release"
wp.init()

@wp.kernel
def compute_transforms(
    shape_body: wp.array(dtype=int),
    shape_transforms: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    num_shapes_per_env: int,
    # outputs
    out_positions: wp.array(dtype=wp.vec3, ndim=2),
    out_rotations: wp.array(dtype=wp.quat, ndim=2),
):
    tid = wp.tid()
    i = shape_body[tid]
    env_id = tid // num_shapes_per_env
    #wp.printf("env_id=%i\n",env_id)

    env_shape_id = tid % num_shapes_per_env
    #wp.printf("env_shape_id=%i\n",env_shape_id)
    X_ws = shape_transforms[i]
    if shape_body:
        body = shape_body[i]
        if body >= 0:
            if body_q:
                X_ws = body_q[body] * X_ws
            else:
                return
    pp = wp.transform_get_translation(X_ws)
    qq = wp.transform_get_rotation(X_ws)
    #wp.printf("pp[%i]=%f %f %f\n", env_id, pp[0],pp[1],pp[2])
    out_rotations[env_id, env_shape_id] = wp.quat(qq[3], qq[0], qq[1], qq[2])
    out_positions[env_id, env_shape_id] = pp



@dataclass
class WarpEnvConfig:

    seed: int = MISSING
    name: str = MISSING
    max_episode_length: int = MISSING
    num_envs: int = MISSING

    # default
    no_grad: bool = True
    visualize: bool = False
    print_env_info: bool = False
    warp_synchronize: bool = True
    device: str = "cuda"
    num_envs: int = 4096

    dt: float = 1.0 / 60.0
    sim_substeps: int = 8
    solve_iterations: int = 2
    joint_linear_relaxation: float = 0.7
    joint_angular_relaxation: float = 0.4
    rigid_contact_relaxation: float = 0.8
    rigid_contact_con_weighting: bool = True
    angular_damping: float = 0.01
    enable_self_collisions: bool = False
    enable_restitution: bool = False
    sim_dt: float = dt


# Base class for all the training environments
# defines warp arrays used for storing observations, reward, actions
# includes standard gym-like functionality like step() function
class WarpEnv(ABC):

    def __init__(self, cfg: dict, num_obs: int, num_acts: int):

        for k, v in cfg.items():
            setattr(self, k, v)

        cpu_madrona = False
        gpu_id = 0
        viz_gpu_hdls = None

        num_worlds = self.num_envs
        self.camera_width = 64
        self.camera_height = 64

        self.madrona = SimManager(
            exec_mode = madrona.ExecMode.CPU if cpu_madrona else madrona.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_worlds,
            max_episode_length = 500,
            enable_batch_renderer = True,
            batch_render_view_width = self.camera_width,
            batch_render_view_height = self.camera_height,
            visualizer_gpu_handles = viz_gpu_hdls,
        )
        self.madrona.init()
        
        self.depth = self.madrona.depth_tensor().to_torch()
        self.rgb = self.madrona.rgb_tensor().to_torch()

        self.madrona_rigid_body_positions = self.madrona.rigid_body_positions_tensor().to_torch()
        self.madrona_rigid_body_rotations = self.madrona.rigid_body_rotations_tensor().to_torch()

        self.step_idx = 0

        #wp.init() moved to global at top of file

        self.integrator = wp.sim.XPBDIntegrator(
            iterations=self.solve_iterations,
            joint_linear_relaxation=self.joint_linear_relaxation,
            joint_angular_relaxation=self.joint_angular_relaxation,
            rigid_contact_relaxation=self.rigid_contact_relaxation,
            rigid_contact_con_weighting=self.rigid_contact_con_weighting,
            angular_damping=self.angular_damping,
            enable_restitution=self.enable_restitution,
        )

        self.articulation_builder = wp.sim.ModelBuilder()

        self.obs_dict = {}
        print("Max episode length = ", self.max_episode_length)

        self.sim_time = 0.0
        self.num_frames = 0  # record the number of frames for rendering
        self.num_agents = 1

        # initialize observation and action space
        self.num_obs = num_obs
        self.num_acts = num_acts

        #self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        
        self.obs_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)

        self.camera_image_stacked =1
        self.camera_channels = 3
        self.num_stacked_channels = self.camera_image_stacked*self.camera_channels
        
        self.madrona_obs_buf = torch.zeros(
                (self.num_envs, self.camera_height, self.camera_width, self.num_stacked_channels), device=self.device, dtype=torch.float)
            

        self.act_space = spaces.Box(np.ones(self.num_acts) * -1.0, np.ones(self.num_acts) * 1.0)

        # allocate buffers
        self.act_buf_wp = wp.zeros(self.num_envs * self.num_acts, dtype=float, device=self.device)
        self.obs_buf_wp = wp.zeros((self.num_envs, self.num_obs), dtype=float, device=self.device)

        self.rew_buf_wp = wp.zeros(self.num_envs, dtype=float, device=self.device)
        self.reset_buf_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.timeout_buf_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.progress_buf_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)

        self.extras = {}

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    # or _step() is a better name?
    @abstractmethod
    def _simulate(self, actions):
        raise NotImplementedError

    def step(self, actions):
        with nvtx.annotate("step", color="orange"):
            # TODO - make clip range a parameter
            actions = torch.clip(actions, -1.0, 1.0)


            self.madrona.process_actions()
            
            self._simulate(actions)

            positions = wp.from_torch(self.madrona_rigid_body_positions, dtype=wp.vec3)
            orientations = wp.from_torch(self.madrona_rigid_body_rotations, dtype=wp.quatf)

            num_shapes_per_env = (self.model.shape_count - 1) // self.num_envs
        
            wp.launch(
            compute_transforms,
            dim=self.model.shape_count,

            inputs=[
                self.model.shape_body,
                self.model.shape_transform,
                self.state_0.body_q,
                num_shapes_per_env,
            ],
            outputs=[
                positions,
                orientations,
            ],
            )

            #wp.synchronize()

            #print("")
            #print(positions)
            #print(orientations)
            #print("")

            #print("positions=",positions)
            #print("orientations=",orientations)

            self.madrona.post_physics()

            self.step_idx += 1

            # copy warp data to pytorch
            self.extras["time_outs"] = (wp.torch.to_torch(self.timeout_buf_wp).to(self.device)).squeeze(-1)
            
            #self.obs_dict["obs"] = wp.torch.to_torch(self.obs_buf_wp).to(self.device)
            
            #todo: copy
            #todo: pass the Madrona image buffer
            self.obs_dict["obs"] = self.madrona_obs_buf

            #print(self.rgb[1, :, :, :3].shape)
            
            #save_image(self.rgb[1, :, :, :3].permute(2, 0, 1).float() / 255, f"out_{self.step_idx}.png")

            #print("self.rgb=",self.rgb)
            #print("self.rgb.shape=",self.rgb.shape)
            rew_buf = (wp.torch.to_torch(self.rew_buf_wp).to(self.device)).squeeze(-1)
            reset_buf = (wp.torch.to_torch(self.reset_buf_wp).to(self.device)).squeeze(-1)
            
            return self.obs_dict, rew_buf, reset_buf, self.extras

    def get_state(self):
        return self.state_0.joint_q.clone(), self.state_0.joint_qd.clone()

    def reset_with_state(self, init_joint_q, init_joint_qd, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # fixed start state
            self.state_0.joint_q = self.state_0.joint_q.clone()
            self.state_0.joint_qd = self.state_0.joint_qd.clone()
            self.state_0.joint_q.view(self.num_envs, -1)[env_ids, :] = init_joint_q.view(-1, self.num_joint_q)[
                env_ids, :
            ].clone()
            self.state_0.joint_qd.view(self.num_envs, -1)[env_ids, :] = init_joint_qd.view(-1, self.num_joint_qd)[
                env_ids, :
            ].clone()

            self.progress_buf[env_ids] = 0

            self.calculateObservations()
            

        return self.obs_buf

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def compute_env_offsets(self, env_offset=(5.0, 0.0, 5.0), upaxis="y"):
        # compute positional offsets per environment (used for visualization)
        nonzeros = np.nonzero(env_offset)[0]
        env_offset = np.array(env_offset)
        num_dim = nonzeros.shape[0]
        if num_dim > 0:
            side_length = int(np.ceil(self.num_envs ** (1.0 / num_dim)))
            env_offsets = []
        else:
            env_offsets = np.zeros((self.num_envs, 3))
        if num_dim == 1:
            for i in range(self.num_envs):
                env_offsets.append(i * env_offset)
        elif num_dim == 2:
            for i in range(self.num_envs):
                d0 = i // side_length
                d1 = i % side_length
                offset = np.zeros(3)
                offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
                offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
                env_offsets.append(offset)
        elif num_dim == 3:
            for i in range(self.num_envs):
                d0 = i // (side_length * side_length)
                d1 = (i // side_length) % side_length
                d2 = i % side_length
                offset = np.zeros(3)
                offset[0] = d0 * env_offset[0]
                offset[1] = d1 * env_offset[1]
                offset[2] = d2 * env_offset[2]
                env_offsets.append(offset)
        env_offsets = np.array(env_offsets)
        min_offsets = np.min(env_offsets, axis=0)
        correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
        if isinstance(upaxis, str):
            upaxis = "xyz".index(upaxis.lower())
        correction[upaxis] = 0.0  # ensure the envs are not shifted below the ground plane
        env_offsets -= correction
        return env_offsets



np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")


@dataclass
class CartpoleCameraConfig(WarpEnvConfig):
    name: str = "CartpoleCamera"
    stochastic_init: bool = True
    max_episode_length: int = 256  # 240

    action_strength: float = 1500.0
    sim_substeps: int = 4
    solve_iterations: int = 2


class CartpoleCamera(WarpEnv):

    # add Madrona here
    # modify observation
    # use a cnn or stack 2 images (so velocity can be inferred)

    def __init__(self, cfg: WarpEnvConfig):

        num_obs = 5
        num_act = 1

        super(CartpoleCamera, self).__init__(cfg, num_obs, num_act)

        self.init_sim()
        self.create_graph()

        if self.print_env_info:
            num_joint_act = int(len(self.model.joint_act) / self.num_envs)

            print(self.name)
            print("Num envs: ", cfg["num_envs"])
            print("Stochastic init: ", cfg["stochastic_init"])
            print("Num dofs: ", int(len(self.model.joint_act) / self.num_envs))
            print("Num joint actions: ", num_joint_act)

        self.joint_act = to_torch(self.model.joint_act)

        # -----------------------
        # set up Usd renderer
        print("self.visualize=",self.visualize)
        if self.visualize:
            self.stage = "outputs/" + "CartpoleCameraWarp_NoCopy_" + str(self.num_envs) + ".usd"

            # self.renderer = wp.sim.render.SimRenderer(self.model, self.stage)
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, self.stage, vsync=False)
            self.render_time = 0.0

    def init_sim(self):
        self.builder = wp.sim.ModelBuilder()

        self.ground = False

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        self.inv_start_rot_wp = wp.quat_inverse(self.start_rot)

        self.start_pos = []
        # apply some non-zero angle to the pole
        self.start_joint_q = [0.0, 0.05]

        if self.visualize:
            self.env_dist = 4.0
        else:
            self.env_dist = 0.0  # set to zero for training for numerical consistency

        wp.sim.parse_urdf(
            "assets/cartpole_dflex.urdf",
            self.articulation_builder,
            floating=False,
            density=1000.0,
            armature=0.0,
            stiffness=0.0,
            damping=0.01,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            ensure_nonstatic_links=False,
            enable_self_collisions=False,
        )

        self.articulation_builder.collapse_fixed_joints()

        self.num_q = len(self.articulation_builder.joint_q)
        self.num_qd = len(self.articulation_builder.joint_qd)
        self.num_bodies = len(self.articulation_builder.body_q)

        if self.visualize:
            env_offsets = self.compute_env_offsets(env_offset=(6.0, 0.0, 2.0))

        for i in range(self.num_envs):

            # base transform
            if self.visualize:
                self.start_pos.append(wp.vec3(env_offsets[i]))
            else:
                start_pos_x = 4.0 + i * self.env_dist
                start_pos_y = 4.0
                start_pos_z = 0.0
                self.start_pos.append(wp.vec3([start_pos_x, start_pos_y, start_pos_z]))

            xform = wp.transform(self.start_pos[-1], self.start_rot)
            self.builder.add_builder(self.articulation_builder, xform=xform)

            self.builder.joint_q[i * self.num_q : (i + 1) * self.num_q] = self.start_joint_q

        self.start_pos_wp = wp.array(self.start_pos, dtype=wp.vec3, device=self.device)

        self.joint_start_q_np = np.asarray(self.builder.joint_q)
        self.joint_start_qd_np = np.asarray(self.builder.joint_qd)

        if self.stochastic_init:
            # joint_q_rand = 2.0 * np.random.rand(*self.joint_start_q_np.shape) - 1.0
            joint_q_rand = 2.0 * np.random.rand(self.num_envs, self.num_q) - 1.0
            joint_q_rand[:, 0] *= 0.5
            joint_q_rand[:, 1] *= 0.5  # np.pi # uncomment for upside down
            joint_q_rand = joint_q_rand.reshape(*self.joint_start_q_np.shape)

            joint_qd_rand = 2.0 * np.random.rand(*self.joint_start_qd_np.shape) - 1.0
            self.joint_start_qd_np = 0.5 * joint_qd_rand

            for i in range(self.num_envs):
                self.joint_start_q_np[i * self.num_q : (i + 1) * self.num_q] = joint_q_rand[
                    i * self.num_q : (i + 1) * self.num_q
                ]

        self.joint_limit_lower_np = self.builder.joint_limit_lower
        self.joint_limit_upper_np = self.builder.joint_limit_upper

        # finalize model
        self.builder.num_envs = self.num_envs
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground

        self.joint_start_q = wp.array(self.joint_start_q_np, dtype=wp.float32, device=self.device)
        self.joint_start_qd = wp.array(self.joint_start_qd_np, dtype=wp.float32, device=self.device)

        self.joint_limit_lower = self.model.joint_limit_lower
        self.joint_limit_upper = self.model.joint_limit_upper

        self.joint_rand_q = wp.zeros_like(self.model.joint_q)
        self.joint_rand_qd = wp.zeros_like(self.model.joint_qd)

        self.q_ik = wp.zeros_like(self.model.joint_q)
        self.qd_ik = wp.zeros_like(self.model.joint_qd)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # set body state
        wp.sim.eval_fk(self.model, self.joint_start_q, self.joint_start_qd, None, self.state_0)

    def create_graph(self):
        # create update graph
        wp.capture_begin()

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt / self.sim_substeps)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.sim.eval_ik(self.model, self.state_0, self.q_ik, self.qd_ik)

        wp.launch(
            kernel=calculate_observations,
            dim=self.num_envs,
            inputs=[self.q_ik, self.qd_ik, self.obs_buf_wp, self.num_obs, self.num_q, self.num_qd],
            device=self.device,
        )

        wp.launch(
            kernel=calculate_reward,
            dim=self.num_envs,
            inputs=[
                self.rew_buf_wp,
                self.reset_buf_wp,
                self.timeout_buf_wp,
                self.progress_buf_wp,
                self.max_episode_length,
                self.q_ik,
                self.qd_ik,
                self.obs_buf_wp,
                self.num_obs,
                self.num_q,
                self.num_qd,
                self.act_buf_wp,
                self.num_acts,
            ],
            device=self.device,
        )

        if self.stochastic_init:
            # random reset
            wp.launch(
                kernel=randomize_start_state,
                dim=self.num_envs,
                inputs=[
                    self.seed,
                    self.joint_start_q,
                    self.joint_limit_lower,
                    self.joint_limit_upper,
                    self.joint_rand_q,
                    self.joint_rand_qd,
                    self.num_q,
                    self.num_qd,
                    self.reset_buf_wp,
                ],
                device=self.device,
            )

            wp.sim.eval_fk(self.model, self.joint_rand_q, self.joint_rand_qd, self.reset_buf_wp, self.state_0)
        else:
            wp.sim.eval_fk(self.model, self.joint_start_q, self.joint_start_qd, self.reset_buf_wp, self.state_0)

        wp.sim.eval_ik(self.model, self.state_0, self.q_ik, self.qd_ik)

        wp.launch(
            kernel=calculate_observations,
            dim=self.num_envs,
            inputs=[self.q_ik, self.qd_ik, self.obs_buf_wp, self.num_obs, self.num_q, self.num_qd],
            device=self.device,
        )

        self.graph = wp.capture_end()

    def render(self, mode="human"):
        if self.visualize:
            self.render_time += self.dt

            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

            # render_interval = 1
            # if (self.num_frames == render_interval):
            #     try:
            #         self.renderer.save()
            #     except:
            #         print("USD save error")

            #     self.num_frames -= render_interval

    def _simulate(self, actions):

        with nvtx.annotate("_simulate", color="green"):
            self.joint_act.view(self.num_envs, -1)[:, 0:1] = actions * self.action_strength

            wp.capture_launch(self.graph)

            if self.warp_synchronize:
                wp.synchronize()

            self.sim_time += self.sim_dt
            self.num_frames += 1

            # if self.no_grad == False:
            #     self.obs_buf_before_reset = self.obs_buf.clone()
            #     self.extras = {
            #         'obs_before_reset': self.obs_buf_before_reset,
            #         'episode_end': self.termination_buf
            #         }

            self.render()

    def reset(self):
        self.reset_buf_wp.fill_(1)

        if self.stochastic_init:
            # random reset
            wp.launch(
                kernel=randomize_start_state,
                dim=self.num_envs,
                inputs=[
                    self.seed,
                    self.joint_start_q,
                    self.joint_limit_lower,
                    self.joint_limit_upper,
                    self.joint_rand_q,
                    self.joint_rand_qd,
                    self.num_q,
                    self.num_qd,
                    self.reset_buf_wp,
                ],
                device=self.device,
            )

            wp.sim.eval_fk(self.model, self.joint_rand_q, self.joint_rand_qd, self.reset_buf_wp, self.state_0)
        else:
            wp.sim.eval_fk(self.model, self.joint_start_q, self.joint_start_qd, self.reset_buf_wp, self.state_0)

        wp.sim.eval_ik(self.model, self.state_0, self.q_ik, self.qd_ik)

        wp.launch(
            kernel=calculate_observations,
            dim=self.num_envs,
            inputs=[self.q_ik, self.qd_ik, self.obs_buf_wp, self.num_obs, self.num_q, self.num_qd],
            device=self.device,
        )

        #self.obs_dict["obs"] = to_torch(self.obs_buf_wp).view(self.num_envs, -1).to(self.device).clone()
        self.obs_dict["obs"] = self.madrona_obs_buf.clone()
        
        self.reset_buf_wp.fill_(0)

        return self.obs_dict


@wp.kernel
def calculate_reward(
    rew_buf: wp.array(dtype=float),
    reset_buf: wp.array(dtype=int),
    timeout_buf: wp.array(dtype=int),
    progress_buf: wp.array(dtype=int),
    max_episode_length: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    obs_buf: wp.array(dtype=float, ndim=2),
    num_obs: int,
    num_q: int,
    num_qd: int,
    actions: wp.array(dtype=float),
    num_act: int,
):

    tid = wp.tid()

    timeout_buf[tid] = 0
    reset_buf[tid] = 0
    progress_buf[tid] = progress_buf[tid] + 1

    q_offset = tid * num_q
    qd_offset = tid * num_qd
    act_offset = tid * num_act

    x = joint_q[q_offset + 0]
    theta = joint_q[q_offset + 1]
    x_dot = joint_qd[qd_offset + 0]
    theta_dot = joint_qd[qd_offset + 1]

    pole_angle_penalty = 10.0
    pole_velocity_penalty = 0.1

    cart_position_penalty = 0.05
    cart_velocity_penalty = 0.1

    cart_action_penalty = 0.0

    action_mag_sq = float(0.0)
    for i in range(num_act):
        action_mag_sq += actions[act_offset + i] ** 2.0

    rew_buf[tid] = (
        # For now wp.pow is not working
        # -wp.pow(theta, 2.) * pole_angle_penalty \
        # -wp.pow(theta_dot, 2.) * pole_velocity_penalty \
        # -wp.pow(x, 2.) * cart_position_penalty \
        # -wp.pow(x_dot, 2.) * cart_velocity_penalty \
        -theta * theta * pole_angle_penalty
        - theta_dot * theta_dot * pole_velocity_penalty
        #- x * x * cart_position_penalty
        #- x_dot * x_dot * cart_velocity_penalty
        - action_mag_sq * cart_action_penalty
    )

    # reset agents
    if progress_buf[tid] >= max_episode_length:
        reset_buf[tid] = 1
        timeout_buf[tid] = 1
        progress_buf[tid] = 0


@wp.kernel
def randomize_start_state(
    seed: int,
    joint_q: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    rand_joint_q: wp.array(dtype=float),
    rand_joint_qd: wp.array(dtype=float),
    num_q: int,
    num_qd: int,
    reset_buf: wp.array(dtype=int),
):

    tid = wp.tid()

    if reset_buf[tid] != 1:
        return

    state = wp.rand_init(seed, tid)

    offset = num_q * tid

    rand_q_range = 0.05
    rand_qd_range = 0.1

    for i in range(num_qd):
        q_offset = offset + i

        # rand_joint_q[q_offset] = wp.randf(state, joint_limit_lower[q_offset], joint_limit_upper[q_offset])

        rand_joint_q[q_offset] = wp.randf(state, -rand_q_range, rand_q_range)
        # rand_joint_q[q_offset] = joint_q[q_offset] + rand_joint_q[q_offset]
        # rand_joint_q[q_offset] = wp.clamp(rand_joint_q[q_offset], joint_limit_lower[q_offset], joint_limit_upper[q_offset])

        rand_joint_qd[q_offset] = wp.randf(state, -rand_qd_range, rand_qd_range)


@wp.kernel
def calculate_observations(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    obs_buf: wp.array(dtype=float, ndim=2),
    num_obs: int,
    num_q: int,
    num_qd: int,
):

    tid = wp.tid()

    q_offset = tid * num_q
    qd_offset = tid * num_qd

    x = joint_q[q_offset + 0]
    theta = joint_q[q_offset + 1]
    x_dot = joint_qd[qd_offset + 0]
    theta_dot = joint_qd[qd_offset + 1]

    obs_buf[tid][0] = x
    obs_buf[tid][1] = x_dot
    obs_buf[tid][2] = wp.sin(theta)
    obs_buf[tid][3] = wp.cos(theta)
    obs_buf[tid][4] = theta_dot


