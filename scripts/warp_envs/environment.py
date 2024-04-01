# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from enum import Enum
from typing import Tuple

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from tqdm import trange

wp.init()


class RenderMode(Enum):
    NONE = "none"
    OPENGL = "opengl"
    USD = "usd"

    def __str__(self):
        return self.value


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    FEATHERSTONE = "featherstone"

    def __str__(self):
        return self.value


def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"):
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
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
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets


@wp.kernel
def assign_joint_q_qd_obs(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    obs: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    for i in range(dof_q_per_env):
        obs[tid, i] = joint_q[tid * dof_q_per_env + i]
    for i in range(dof_qd_per_env):
        obs[tid, i + dof_q_per_env] = joint_qd[tid * dof_qd_per_env + i]


class Environment:
    sim_name: str = "Environment"

    frame_dt = 1.0 / 60.0

    episode_duration = 5.0  # seconds
    episode_frames = None  # number of steps per episode, if None, use episode_duration / frame_dt

    # whether to play the simulation indefinitely when using the OpenGL renderer
    continuous_opengl_render: bool = True

    sim_substeps_euler: int = 16
    sim_substeps_xpbd: int = 5

    euler_settings = dict()
    xpbd_settings = dict()

    render_mode: RenderMode = RenderMode.OPENGL
    opengl_render_settings = dict()
    usd_render_settings = dict(scaling=10.0)
    show_rigid_contact_points = False
    contact_points_radius = 1e-3
    show_joints = False
    # whether OpenGLRenderer should render each environment in a separate tile
    use_tiled_rendering = False

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True

    use_graph_capture = False #bool = wp.get_preferred_device().is_cuda

    activate_ground_plane: bool = True

    integrator_type: IntegratorType = IntegratorType.XPBD

    up_axis: str = "Y"
    gravity: float = -9.81

    # stiffness and damping for joint attachment dynamics used by Euler
    joint_attach_ke: float = 32000.0
    joint_attach_kd: float = 50.0

    # maximum number of rigid contact points to generate per mesh
    rigid_mesh_contact_max: int = 0  # (0 = unlimited)

    # distance threshold at which contacts are generated
    rigid_contact_margin: float = 0.05
    # whether to iterate over mesh vertices for box/capsule collision
    rigid_contact_iterate_mesh_vertices: bool = True

    # number of search iterations for finding closest contact points between edges and SDF
    edge_sdf_iter: int = 10

    # whether each environment should have its own collision group
    # to avoid collisions between environments
    separate_collision_group_per_env: bool = True

    plot_body_coords: bool = False
    plot_joint_coords: bool = False

    custom_dynamics = None

    # control-related definitions, to be updated by derived classes
    controllable_dofs = []
    control_gains = []
    control_limits = []

    def __init__(
        self,
        num_envs: int = 100,
        episode_frames: int = None,
        integrator_type: IntegratorType = None,
        render_mode: RenderMode = None,
        env_offset: Tuple[float, float, float] = (1.0, 0.0, 1.0),
        device: wp.context.Devicelike = None,
        requires_grad: bool = False,
        profile: bool = False,
        enable_timers: bool = False,
        use_graph_capture: bool = None,
    ):
        self.num_envs = num_envs
        if episode_frames is not None:
            self.episode_frames = episode_frames
        if integrator_type is not None:
            self.integrator_type = integrator_type
        if render_mode is not None:
            self.render_mode = render_mode
        if use_graph_capture is not None:
            self.use_graph_capture = use_graph_capture
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad
        self.profile = profile
        self.enable_timers = enable_timers

        if self.use_tiled_rendering and self.render_mode == RenderMode.OPENGL:
            # no environment offset when using tiled rendering
            self.env_offset = (0.0, 0.0, 0.0)
        else:
            self.env_offset = env_offset

        if isinstance(self.up_axis, str):
            up_vector = np.zeros(3)
            up_vector["xyz".index(self.up_axis.lower())] = 1.0
        else:
            up_vector = self.up_axis
        builder = wp.sim.ModelBuilder(up_vector=up_vector, gravity=self.gravity)
        builder.rigid_mesh_contact_max = self.rigid_mesh_contact_max
        builder.rigid_contact_margin = self.rigid_contact_margin
        self.env_offsets = compute_env_offsets(self.num_envs, self.env_offset, self.up_axis)
        try:
            articulation_builder = wp.sim.ModelBuilder(up_vector=up_vector, gravity=self.gravity)
            self.create_articulation(articulation_builder)
            for i in trange(self.num_envs, desc=f"Creating {self.num_envs} environments"):
                xform = wp.transform(self.env_offsets[i], wp.quat_identity())
                builder.add_builder(
                    articulation_builder, xform, separate_collision_group=self.separate_collision_group_per_env
                )
            self.bodies_per_env = len(articulation_builder.body_q)
            self.dof_q_per_env = articulation_builder.joint_coord_count
            self.dof_qd_per_env = articulation_builder.joint_dof_count
        except NotImplementedError:
            # custom simulation setup where something other than an articulation is used
            self.setup(builder)
            self.bodies_per_env = len(builder.body_q)
            self.dof_q_per_env = builder.joint_coord_count
            self.dof_qd_per_env = builder.joint_dof_count

        self.model = builder.finalize(requires_grad=self.requires_grad)
        self.customize_model(self.model)
        self.device = self.model.device
        if not self.model.device.is_cuda:
            self.use_graph_capture = False
        self.model.ground = self.activate_ground_plane

        self.model.joint_attach_ke = self.joint_attach_ke
        self.model.joint_attach_kd = self.joint_attach_kd

        if self.integrator_type == IntegratorType.EULER:
            self.sim_substeps = self.sim_substeps_euler
            self.integrator = wp.sim.SemiImplicitIntegrator(**self.euler_settings)
        elif self.integrator_type == IntegratorType.XPBD:
            self.sim_substeps = self.sim_substeps_xpbd
            self.integrator = wp.sim.XPBDIntegrator(**self.xpbd_settings)
        elif self.integrator_type == IntegratorType.FEATHERSTONE:
            self.sim_substeps = self.sim_substeps_euler
            self.integrator = wp.sim.FeatherstoneIntegrator(self.model, **self.euler_settings)

        if self.episode_frames is None:
            self.episode_frames = int(self.episode_duration / self.frame_dt)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = self.episode_frames * self.sim_substeps
        self.sim_step = 0
        self.sim_time = 0.0
        self.invalidate_cuda_graph = False

        self.controls = []
        # for _ in range(self.sim_steps):
        for _ in range(1):
            control = self.model.control()
            self.customize_control(control)
            self.controls.append(control)

        if self.requires_grad:
            self.states = []
            for _ in range(self.sim_steps + 1):
                state = self.model.state()
                self.customize_state(state)
                self.states.append(state)
            self.update = self.update_grad
        else:
            # set up current and next state to be used by the integrator
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.customize_state(self.state_0)
            self.customize_state(self.state_1)
            self.update = self.update_nograd
            if self.use_graph_capture:
                self.state_temp = self.model.state()
            else:
                self.state_temp = None

        self.renderer = None
        if self.profile:
            self.render_mode = RenderMode.NONE
        if self.render_mode == RenderMode.OPENGL:
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model,
                self.sim_name,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                contact_points_radius=self.contact_points_radius,
                show_joints=self.show_joints,
                **self.opengl_render_settings,
            )
            if self.use_tiled_rendering and self.num_envs > 1:
                floor_id = self.model.shape_count - 1
                # all shapes except the floor
                instance_ids = np.arange(floor_id, dtype=np.int32).tolist()
                shapes_per_env = floor_id // self.num_envs
                additional_instances = []
                if self.activate_ground_plane:
                    additional_instances.append(floor_id)
                self.renderer.setup_tiled_rendering(
                    instances=[
                        instance_ids[i * shapes_per_env : (i + 1) * shapes_per_env] + additional_instances
                        for i in range(self.num_envs)
                    ]
                )
        elif self.render_mode == RenderMode.USD:
            filename = os.path.join(os.path.dirname(__file__), "..", "outputs", self.sim_name + ".usd")
            self.renderer = wp.sim.render.SimRendererUsd(
                self.model,
                filename,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                **self.usd_render_settings,
            )

    @property
    def uses_generalized_coordinates(self):
        # whether the model uses generalized or maximal coordinates (joint q/qd vs body q/qd) in the state
        return self.integrator_type == IntegratorType.FEATHERSTONE

    def create_articulation(self, builder):
        raise NotImplementedError

    def setup(self, builder):
        pass

    def before_simulate(self):
        pass

    def after_simulate(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def before_update(self):
        pass

    def after_update(self):
        pass

    def custom_render(self, render_state, renderer):
        pass

    @property
    def state(self):
        # shortcut to current state
        if self.requires_grad:
            return self.states[self.sim_step]
        return self.state_0

    @property
    def next_state(self):
        # shortcut to subsequent state
        if self.requires_grad:
            return self.states[self.sim_step + 1]
        return self.state_1

    @property
    def control(self):
        return self.controls[min(len(self.controls) - 1, max(0, (self.sim_step - 1) % self.sim_steps))]

    @property
    def control_input(self):
        # points to the actuation input of the control
        return self.control.joint_act

    def customize_state(self, state: wp.sim.State):
        pass

    def customize_control(self, control: wp.sim.Control):
        pass

    def customize_model(self, model):
        pass

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        pass

    @property
    def control_dim(self):
        return len(self.controllable_dofs)

    @property
    def observation_dim(self):
        # default observation consists of generalized joint positions and velocities
        return self.dof_q_per_env + self.dof_qd_per_env

    def compute_observations(
        self, state: wp.sim.State, control: wp.sim.Control, observations: wp.array, step: int, horizon_length: int
    ):
        if not self.uses_generalized_coordinates:
            # evaluate generalized coordinates
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        wp.launch(
            assign_joint_q_qd_obs,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, self.dof_q_per_env, self.dof_qd_per_env],
            outputs=[observations],
            device=self.device,
        )

    def update_nograd(self):
        self.before_update()
        if self.use_graph_capture:
            state_0_dict = self.state_0.__dict__
            state_1_dict = self.state_1.__dict__
            state_temp_dict = self.state_temp.__dict__ if self.state_temp is not None else None
        for i in range(self.sim_substeps):
            self.before_step()
            if self.custom_dynamics is not None:
                self.custom_dynamics(self.model, self.state_0, self.state_1, self.sim_dt, self.control)
            else:
                self.state_0.clear_forces()
                with wp.ScopedTimer("collision_handling", color="orange", active=self.enable_timers):
                    wp.sim.collide(
                        self.model,
                        self.state_0,
                        edge_sdf_iter=self.edge_sdf_iter,
                        iterate_mesh_vertices=self.rigid_contact_iterate_mesh_vertices,
                    )
                with wp.ScopedTimer("simulation", color="red", active=self.enable_timers):
                    self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt, self.control)
            self.after_step()
            if i < self.sim_substeps - 1 or not self.use_graph_capture:
                # we can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0
            elif self.use_graph_capture:
                assert (
                    hasattr(self, "state_temp") and self.state_temp is not None
                ), "state_temp must be allocated when using graph capture"
                # swap states by actually copying the state arrays to make sure the graph capture works
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])
            self.sim_time += self.sim_dt
            self.sim_step += 1
        self.after_update()

    def update_grad(self):
        self.before_update()
        for i in range(self.sim_substeps):
            self.before_step()
            if self.custom_dynamics is not None:
                self.custom_dynamics(
                    self.model,
                    self.states[self.sim_step],
                    self.states[self.sim_step + 1],
                    self.sim_dt,
                    self.control,
                )
            else:
                self.states[self.sim_step].clear_forces()
                wp.sim.collide(
                    self.model,
                    self.states[self.sim_step],
                    edge_sdf_iter=self.edge_sdf_iter,
                    iterate_mesh_vertices=self.rigid_contact_iterate_mesh_vertices,
                )
                self.integrator.simulate(
                    self.model,
                    self.states[self.sim_step],
                    self.states[self.sim_step + 1],
                    self.sim_dt,
                    self.control,
                )
            self.after_step()
            self.sim_time += self.sim_dt
            self.sim_step += 1
        self.after_update()

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", color="yellow", active=self.enable_timers):
                self.renderer.begin_frame(self.sim_time)
                # render state 1 (swapped with state 0 just before)
                if self.requires_grad:
                    # ensure we do not render beyond the last state
                    # render_state = state or self.states[min(self.sim_steps, self.sim_step + 1)]
                    render_state = state or self.states[min(self.sim_steps, self.sim_step)]
                else:
                    render_state = state or self.next_state
                with wp.ScopedTimer("custom_render", color="orange", active=self.enable_timers):
                    self.custom_render(render_state, renderer=self.renderer)
                self.renderer.render(render_state)
                self.renderer.end_frame()

    def reset(self):
        self.sim_time = 0.0
        self.sim_step = 0

        if self.eval_fk:
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        if self.model.particle_count > 1:
            self.model.particle_grid.build(
                self.state.particle_q,
                self.model.particle_max_radius * 2.0,
            )

    def step(self):
        # ---------------
        # step simulation

        if self.use_graph_capture:
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
        else:
            self.update()
            self.sim_time += self.frame_dt

    def run(self):
        # ---------------
        # run simulation
        self.reset()

        self.before_simulate()

        if self.renderer is not None:
            self.render(self.state)

            if self.render_mode == RenderMode.OPENGL:
                self.renderer.paused = True

        profiler = {}

        if self.use_graph_capture:
            with wp.ScopedCapture() as capture:
                self.update()
            graph = capture.graph

        if self.plot_body_coords:
            q_history = []
            q_history.append(self.state.body_q.numpy().copy())
            qd_history = []
            qd_history.append(self.state.body_qd.numpy().copy())
            delta_history = []
            delta_history.append(self.state.body_deltas.numpy().copy())
            num_con_history = []
            num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())
        if self.plot_joint_coords:
            joint_q_history = []
            joint_q = wp.zeros_like(self.model.joint_q)
            joint_qd = wp.zeros_like(self.model.joint_qd)

        # simulate
        with wp.ScopedTimer(
            "run_loop", detailed=False, print=False, active=self.enable_timers or self.profile, dict=profiler
        ):
            running = True
            while running:
                for f in range(self.episode_frames):
                    if self.model.particle_count > 1:
                        self.model.particle_grid.build(
                            self.state.particle_q,
                            self.model.particle_max_radius * 2.0,
                        )
                    if self.use_graph_capture:
                        with wp.ScopedTimer("simulation_graph", color="green", active=self.enable_timers):
                            wp.capture_launch(graph)
                        self.sim_time += self.frame_dt
                        self.sim_step += self.sim_substeps
                    elif not self.requires_grad or self.sim_step < self.sim_steps:
                        with wp.ScopedTimer("update", color="green", active=self.enable_timers):
                            self.update()

                        if not self.profile:
                            if self.plot_body_coords:
                                q_history.append(self.state.body_q.numpy().copy())
                                qd_history.append(self.state.body_qd.numpy().copy())
                                delta_history.append(self.state.body_deltas.numpy().copy())
                                num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

                            if self.plot_joint_coords:
                                wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                                joint_q_history.append(joint_q.numpy().copy())

                    self.render()
                    if self.render_mode == RenderMode.OPENGL and self.renderer.has_exit:
                        running = False
                        break

                if not self.continuous_opengl_render or self.render_mode != RenderMode.OPENGL:
                    break

            wp.synchronize()

        self.after_simulate()

        avg_time = np.array(profiler["run_loop"]).mean() / self.episode_frames
        avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if self.renderer is not None:
            self.renderer.save()

        if self.plot_body_coords:
            import matplotlib.pyplot as plt

            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)

            # find bodies with non-zero mass
            body_indices = np.where(self.model.body_mass.numpy() > 0)[0]
            body_indices = body_indices[:5]  # limit number of bodies to plot

            fig, ax = plt.subplots(len(body_indices), 7, figsize=(10, 10), squeeze=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i, 0].set_title(f"Body {j} Position")
                ax[i, 0].grid()
                ax[i, 1].set_title(f"Body {j} Orientation")
                ax[i, 1].grid()
                ax[i, 2].set_title(f"Body {j} Linear Velocity")
                ax[i, 2].grid()
                ax[i, 3].set_title(f"Body {j} Angular Velocity")
                ax[i, 3].grid()
                ax[i, 4].set_title(f"Body {j} Linear Delta")
                ax[i, 4].grid()
                ax[i, 5].set_title(f"Body {j} Angular Delta")
                ax[i, 5].grid()
                ax[i, 6].set_title(f"Body {j} Num Contacts")
                ax[i, 6].grid()
                ax[i, 0].plot(q_history[:, j, :3])
                ax[i, 1].plot(q_history[:, j, 3:])
                ax[i, 2].plot(qd_history[:, j, 3:])
                ax[i, 3].plot(qd_history[:, j, :3])
                ax[i, 4].plot(delta_history[:, j, 3:])
                ax[i, 5].plot(delta_history[:, j, :3])
                ax[i, 6].plot(num_con_history[:, j])
                ax[i, 0].set_xlim(0, self.sim_steps)
                ax[i, 1].set_xlim(0, self.sim_steps)
                ax[i, 2].set_xlim(0, self.sim_steps)
                ax[i, 3].set_xlim(0, self.sim_steps)
                ax[i, 4].set_xlim(0, self.sim_steps)
                ax[i, 5].set_xlim(0, self.sim_steps)
                ax[i, 6].set_xlim(0, self.sim_steps)
                ax[i, 6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if self.plot_joint_coords and len(joint_q_history) > 0:
            import matplotlib.pyplot as plt

            joint_q_history = np.array(joint_q_history)
            dof_q = joint_q_history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof_q)))
            nrows = int(np.ceil(dof_q / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )

            joint_id = 0
            joint_type_names = {
                wp.sim.JOINT_BALL: "ball",
                wp.sim.JOINT_REVOLUTE: "hinge",
                wp.sim.JOINT_PRISMATIC: "slide",
                wp.sim.JOINT_UNIVERSAL: "universal",
                wp.sim.JOINT_COMPOUND: "compound",
                wp.sim.JOINT_FREE: "free",
                wp.sim.JOINT_FIXED: "fixed",
                wp.sim.JOINT_DISTANCE: "distance",
                wp.sim.JOINT_D6: "D6",
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while joint_id < len(joint_type) - 1 and joint_type[joint_id] == wp.sim.JOINT_FIXED:
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            qd_i = qd_start[joint_id]
            num_joint_free = 0  # needed to offset joint_limits, which skip free joint type
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof_q:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(joint_q_history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE:
                    lower = joint_lower[qd_i - num_joint_free]
                    if abs(lower) < 2 * np.pi:
                        ax.axhline(lower, color="red")
                    upper = joint_upper[qd_i - num_joint_free]
                    if abs(upper) < 2 * np.pi:
                        ax.axhline(upper, color="red")
                else:
                    num_joint_free += 1
                joint_name = joint_type_names[joint_type[joint_id]]
                ax.set_title(f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})")
                if joint_id < self.model.joint_count - 1 and q_start[joint_id + 1] == dim + 1:
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()

        return 1000.0 * float(self.num_envs) / avg_time


def run_env(Environment):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=IntegratorType.XPBD.value,
    )
    parser.add_argument(
        "--visualizer",
        help="Type of renderer",
        type=RenderMode,
        choices=list(RenderMode),
        default=RenderMode.OPENGL.value,
    )
    parser.add_argument("--num_envs", help="Number of environments to simulate", type=int, default=100)
    parser.add_argument("--profile", help="Enable profiling", type=bool, default=False)
    args = parser.parse_args()
    env_args = dict(
        integrator_type=args.integrator,
        render_mode=args.visualizer,
        profile=args.profile,
    )

    if args.profile:
        import matplotlib.pyplot as plt

        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            demo = Environment(
                num_envs=env_count,
                **env_args,
            )
            steps_per_second = demo.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

        # plot
        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()
    else:
        demo = Environment(
            num_envs=args.num_envs,
            **env_args,
        )
        return demo.run()
