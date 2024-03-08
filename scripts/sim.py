from madrona_warp_proto_sim import SimManager, madrona
from warp_envs.env_cartpole import CartpoleEnvironment
from warp_envs.environment import RenderMode
import warp as wp
wp.init()

@wp.kernel
def compute_transforms(
    shape_body: wp.array(dtype=int),
    shape_transforms: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    out_positions: wp.array(dtype=wp.vec3),
    out_rotations: wp.array(dtype=wp.quat),
):
    tid = wp.tid()
    i = shape_body[tid]
    X_ws = shape_transforms[i]
    if shape_body:
        body = shape_body[i]
        if body >= 0:
            if body_q:
                X_ws = body_q[body] * X_ws
            else:
                return
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    out_positions[tid] = p
    out_rotations[tid] = wp.quat(q[3], q[0], q[1], q[2])

class Simulator:
    def __init__(self, gpu_id, num_worlds, cpu_madrona, viz_gpu_hdls=None):
        # Initialize madrona simulator
        
        CartpoleEnvironment.render_mode = RenderMode.NONE
        CartpoleEnvironment.num_envs = num_worlds
        
        self.env_cartpole = CartpoleEnvironment()
        self.env_cartpole.init()
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
        
        self.depth = self.madrona.depth_tensor().to_torch()
        self.rgb = self.madrona.rgb_tensor().to_torch()

    def step(self):
        self.madrona.process_actions()

        # Warp here
        self.env_cartpole.step()
        
        #print("self.madrona.rigid_body_positions_tensor()=",self.madrona.rigid_body_positions_tensor())
        positions = wp.from_dlpack(self.madrona.rigid_body_positions_tensor())
        #orientations = wp.from_dlpack(self.madrona.rigid_body_rotations_tensor)
        
        #compute_transforms(...)
        #wp.launch(
        #    compute_transforms,
        #    dim=self.env_cartpole.num_envs,
        #    inputs=[
        #        self.env_cartpole.model.body_shape,
        #        self.env_cartpole.model.shape_transform,
        #        self.env_cartpole.state.body_q,
        #    ],
        #    outputs=[
        #        positions,
        #        orientations,
        #    ],
        #)

        #optional
        self.env_cartpole.render()

        self.madrona.post_physics()
