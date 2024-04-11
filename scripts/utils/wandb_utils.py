def get_wandb_hook(cfg):

    def wandb_hook():
        """Hook to setup WandB after the environment has been created."""
        import datetime
        
        if cfg.wandb_activate:
            # Make sure to install WandB if you use this.
            import wandb

            time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                config=cfg_dict,
                sync_tensorboard=True,
                id=f"{cfg.wandb_name}_{time_str}",
                resume="allow",
            )

    return wandb_hook