import wandb


def init_wandb(config):
    wandb.init(
        project="multi-agent-quality-diversity",
        config=config,
        save_code=True,
        name=config.get("experiment_name"),
        mode=config.get("wandb_mode"),
    )
