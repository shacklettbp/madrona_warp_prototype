# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Cartpole environment
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the Environment class.
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math
import warp as wp
import warp.sim

from .environment import Environment, run_env, IntegratorType

@wp.func
def angle_normalize(x: float):
    return ((x + wp.pi) % (2.0 * wp.pi)) - wp.pi


@wp.kernel
def single_cartpole_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    x = joint_q[env_id * 2 + 0]
    th = joint_q[env_id * 2 + 1]
    thdot = joint_qd[env_id * 2 + 1]
    u = joint_act[env_id * 2]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle = angle_normalize(th)
    c = angle**2.0 + 0.1 * x**2.0 + 0.1 * thdot**2.0 + (u * 1e-4) ** 2.0

    wp.atomic_add(cost, env_id, c)

    if terminated:
        terminated[env_id] = abs(angle) > 0.3


@wp.kernel
def double_cartpole_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    th1 = joint_q[env_id * 3 + 1]
    thdot1 = joint_qd[env_id * 3 + 1]
    th2 = joint_q[env_id * 3 + 2]
    thdot2 = joint_qd[env_id * 3 + 2]
    u = joint_act[env_id * 3]

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L270
    angle1 = angle_normalize(th1)
    angle2 = angle_normalize(th2)
    c = angle1**2.0 + 0.1 * thdot1**2.0 + angle2**2.0 + 0.1 * thdot2**2.0 + (u * 1e-4) ** 2.0

    wp.atomic_add(cost, env_id, c)

    if terminated:
        terminated[env_id] = abs(angle1) > 0.3 or abs(angle2) > 0.3


class CartpoleEnvironment(Environment):
    sim_name = "env_cartpole"
    env_offset = (2.0, 0.0, 2.0)
    opengl_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    single_cartpole = True

    sim_substeps_euler = 16
    sim_substeps_xpbd = 5

    activate_ground_plane = False

    # integrator_type = IntegratorType.FEATHERSTONE

    show_joints = True

    controllable_dofs = [0]
    control_gains = [1500.0]
    control_limits = [(-1.0, 1.0)]

    def create_articulation(self, builder):
        if self.single_cartpole:
            path = "cartpole_single.urdf"
        else:
            path = "cartpole.urdf"
            self.opengl_render_settings["camera_pos"] = (40.0, 1.0, 0.0)
            self.opengl_render_settings["camera_front"] = (-1.0, 0.0, 0.0)
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets", path),
            builder,
            xform=wp.transform(
                (0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
            ),
            floating=False,
            armature=0.01,
            stiffness=0.0,
            damping=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # joint initial positions
        if self.single_cartpole:
            builder.joint_q[-2:] = [0.0, 0.1]
        else:
            builder.joint_q[-3:] = [0.0, 0.1, 0.0]

    def compute_cost_termination(
        self,
        state: wp.sim.State,
        control: wp.sim.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if self.integrator_type != IntegratorType.FEATHERSTONE:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        wp.launch(
            single_cartpole_cost if self.single_cartpole else double_cartpole_cost,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, control.joint_act],
            outputs=[cost, terminated],
            device=self.device,
        )


if __name__ == "__main__":
    run_env(CartpoleEnvironment)
