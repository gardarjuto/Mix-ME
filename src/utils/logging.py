import wandb
from wandb.util import generate_id
import os


def init_wandb(config):
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + generate_id()

    wandb.init(
        project="MA-QD",
        config=config,
        save_code=True,
        name=config.get("experiment_name"),
        mode=config.get("wandb_mode"),
        entity="ucl-dark",
    )
