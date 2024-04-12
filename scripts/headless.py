import madrona_warp_proto_sim
import torch
from sim import Simulator
import argparse
from time import time
from torchvision.utils import save_image

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)

arg_parser.add_argument('--cpu-sim', action='store_true')

args = arg_parser.parse_args()

sim = Simulator(args.gpu_id, args.num_worlds, args.cpu_sim)

start = time()

for i in range(args.num_steps):
    sim.step()

    save_image(sim.rgb[1, :, :, :3].permute(2, 0, 1).float() / 255, f"out_{i}.png")


end = time()

print("FPS:", "{:.1f}".format(args.num_steps * args.num_worlds / (end - start)))
