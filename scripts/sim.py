from madrona_warp_proto_sim import SimManager, madrona
from warp_envs.env_cartpole import CartpoleEnvironment
from warp_envs.environment import RenderMode
import warp as wp
import math

wp.init()

@wp.kernel
def compute_transforms(
    shape_body: wp.array(dtype=int),
    shape_transforms: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    num_shapes_per_env: int,
    start_pos_wp: wp.array(dtype=wp.vec3),
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
    
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi/2.0)
    global_tf = wp.transform(wp.vec3(0.,0.,0.), rot)

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
            
    X_ws = global_tf * X_ws

    pp = wp.transform_get_translation(X_ws) - wp.quat_rotate(rot, start_pos_wp[env_id])
    qq = wp.transform_get_rotation(X_ws)
    #wp.printf("pp[%i]=%f %f %f\n", env_id, pp[0],pp[1],pp[2])
    out_rotations[env_id, env_shape_id] = wp.quat(qq[3], qq[0], qq[1], qq[2])
    out_positions[env_id, env_shape_id] = pp


class Simulator:
    def __init__(self, gpu_id, num_worlds, cpu_madrona, viz_gpu_hdls=None):
        # Initialize madrona simulator
        
        #CartpoleEnvironment.render_mode = #RenderMode.NONE
        CartpoleEnvironment.num_envs = num_worlds
        CartpoleEnvironment.env_offset = (0.0, 0.0, 0.0)
        self.env_cartpole = CartpoleEnvironment(num_envs = num_worlds, env_offset = (5.0, 0.0, 5.0), render_mode = RenderMode.OPENGL)
        #self.env_cartpole.init()
        self.env_cartpole.reset()
        
        
        self.madrona = SimManager(
            exec_mode = madrona.ExecMode.CPU if cpu_madrona else madrona.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_worlds,
            max_episode_length = 500,
            enable_batch_renderer = True,
            batch_render_view_width = 64,
            batch_render_view_height = 64,
            visualizer_gpu_handles = viz_gpu_hdls,
        )
        self.madrona.init()
        
        self.start_pos_wp = wp.array(self.env_cartpole.env_offsets, dtype=wp.vec3)
        #print("self.start_pos_wp=",self.start_pos_wp)
        self.depth = self.madrona.depth_tensor().to_torch()
        self.rgb = self.madrona.rgb_tensor().to_torch()

        self.madrona_rigid_body_positions = self.madrona.rigid_body_positions_tensor().to_torch()
        self.madrona_rigid_body_rotations = self.madrona.rigid_body_rotations_tensor().to_torch()

        self.step_idx = 0

    def step(self):
        self.madrona.process_actions()

        # Warp here
        self.env_cartpole.step()
        #wp.sim.eval_fk(self.env_cartpole.model, self.env_cartpole.state.joint_q, self.env_cartpole.state.joint_qd, None, self.env_cartpole.state)
        self.env_cartpole.render()
                
        #print("self.madrona.rigid_body_positions_tensor()=",self.madrona.rigid_body_positions_tensor())
        positions = wp.from_torch(self.madrona_rigid_body_positions, dtype=wp.vec3)
        orientations = wp.from_torch(self.madrona_rigid_body_rotations, dtype=wp.quatf)

        #positions = positions.reshape((positions.shape[0]*positions.shape[1]))
        #print("orientations=",orientations)
        #orientations = orientations.reshape((orientations.shape[0]*orientations.shape[1]))
        #print("positions=",positions)
        #print("positions.shape=", positions.shape)
        #print("orientations=",orientations)
        
        num_shapes_per_env = (self.env_cartpole.model.shape_count - 1) // self.env_cartpole.num_envs
        
        wp.launch(
          compute_transforms,
          dim=self.env_cartpole.model.shape_count,

          inputs=[
              self.env_cartpole.model.shape_body,
              self.env_cartpole.model.shape_transform,
              self.env_cartpole.state.body_q,
              num_shapes_per_env,
              self.start_pos_wp,
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
