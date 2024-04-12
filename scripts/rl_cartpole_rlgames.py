import madrona_warp_proto_sim
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from utils.reformat import omegaconf_to_dict, print_dict
from utils.rlgames_utils import get_rlgames_env_creator, RLGPUEnv, RLGPUAlgoObserver
from utils.wandb_utils import get_wandb_hook

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from utils.load_utils import set_seed


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    
    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        
    # TODO - figure out how to use interpolation for these 
    # unless number of environments provided on command line, default to that in the training config
    if cfg.num_envs != '':
        cfg.train.params.config.num_actors = cfg.num_envs
        cfg.env.num_envs = cfg.num_envs
    else:
        cfg.env.num_envs = cfg.train.params.config.num_actors

    cfg.env.device = cfg.device
    cfg.env.visualize = cfg.visualize
    cfg.env.seed = cfg.seed

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # # set numpy formatting for printing only
    # set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.env),
        post_create_hook=get_wandb_hook(cfg)
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register('WARP',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('warp', {
        'vecenv_type': 'WARP',
        'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
    })


if __name__ == "__main__":
    launch_rlg_hydra()
