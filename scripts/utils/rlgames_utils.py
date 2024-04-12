from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
import torch
import numpy as np
from typing import Callable
from .load_utils import set_seed

#import warp_envs.envs as envs

from hydra.core.config_store import ConfigStore

#from warp_envs.envs.ant import Ant, AntConfig
#from warp_envs.envs.ant_client import AntClient, AntClientConfig
#from rl_cartpole_rlgames import Cartpole, CartpoleConfig
#from rl_cartpole_rlgames import Cartpole, CartpoleConfig

from .env_cartpole_rlgames import Cartpole, CartpoleConfig
from .env_cartpole_camera_rlgames import CartpoleCamera, CartpoleCameraConfig

#from warp_envs.envs.cartpole_client import CartpoleClient, CartpoleClientConfig
#from warp_envs.envs.humanoid import Humanoid, HumanoidConfig
#from warp_envs.envs.humanoid_client import HumanoidClient, HumanoidClientConfig

cs = ConfigStore.instance()
#cs.store(name="Ant", node=AntConfig, group="env")
#cs.store(name="AntClient", node=AntClientConfig, group="env")
#cs.store(name="Humanoid", node=HumanoidConfig, group="env")
#cs.store(name="HumanoidClient", node=HumanoidClientConfig, group="env")
#cs.store(name="Cartpole", node=CartpoleConfig, group="env")
cs.store(name="CartpoleCamera", node=CartpoleCameraConfig, group="env")
#cs.store(name="CartpoleClient", node=CartpoleClientConfig, group="env")



def get_rlgames_env_creator(
        # used to create the vec task
        env_config: dict,
        post_create_hook: Callable = None,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        env_config: environment configuration.
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        # create native task and pass custom config
        #env = getattr(envs, env_config['name'])(
        #    cfg=env_config,
        #)
        env = CartpoleCamera(cfg = env_config)

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                    infotensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, epoch_num)
            self.ep_infos.clear()
        
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)
            self.writer.add_scalar(f'{k}/time', v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, action):
        return  self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        # if self.env.num_states > 0:
        #     info['state_space'] = self.env.state_space
        #     print(info['action_space'], info['observation_space'], info['state_space'])
        # else:
        #     print(info['action_space'], info['observation_space'])

        return info