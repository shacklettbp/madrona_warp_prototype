import torch
from madrona_warp_proto_sim import SimManager, madrona
from madrona_warp_proto_viz import VisualizerGPUState, Visualizer

import numpy as np
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--window-width', type=int, required=True)
arg_parser.add_argument('--window-height', type=int, required=True)

arg_parser.add_argument('--gpu-sim', action='store_true')

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(args.window_width, args.window_height, args.gpu_id)

sim = SimManager(
    exec_mode = madrona.ExecMode.CUDA if args.gpu_sim else madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    max_episode_length = 500,
    enable_batch_renderer = True,
    batch_render_view_width = 64,
    batch_render_view_height = 64,
    visualizer_gpu_handles = viz_gpu_state.get_gpu_handles(),
)
sim.init()

visualizer = Visualizer(viz_gpu_state, sim)

def step_fn():
    sim.process_actions()

    # Warp here

    sim.post_physics()

visualizer.loop(sim, step_fn)
