import wandb
from wandb.util import generate_id
import os


def init_wandb(config):
    wandb.init(
        project=config.get("project_name"),
        config=config,
        save_code=True,
        name=config.get("experiment_name"),
        mode=config.get("wandb_mode"),
        entity=config.get("entity_name"),
    )
