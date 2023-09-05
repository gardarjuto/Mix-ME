import jax
import os
from src.utils.arguments_ippo import (
    load_config,
    parse_arguments,
    merge_configs,
)
from src.utils.logging import init_wandb
from src.training.ippo_mamujoco import make_train

import wandb


def main():
    args = parse_arguments()

    # Load the configuration file and merge it with the CLI arguments
    config = load_config(args.config)
    config = merge_configs(config, args)

    config.update(
        {
            "LR": config["lr"],
            "NUM_ENVS": config["num_envs"],
            "NUM_STEPS": config["num_steps"],
            "TOTAL_TIMESTEPS": config["total_timesteps"],
            "UPDATE_EPOCHS": config["update_epochs"],
            "NUM_MINIBATCHES": config["num_minibatches"],
            "GAMMA": config["gamma"],
            "GAE_LAMBDA": config["gae_lambda"],
            "CLIP_EPS": config["clip_eps"],
            "ENT_COEF": config["ent_coef"],
            "VF_COEF": config["vf_coef"],
            "MAX_GRAD_NORM": config["max_grad_norm"],
            "ACTIVATION": config["activation"],
            "ENV_NAME": config["env_name"],
            "ENV_KWARGS": {"homogenisation_method": "max"},
            "ANNEAL_LR": config["anneal_lr"],
        }
    )

    # Initialise WandB
    init_wandb(config)

    config = wandb.config

    if config["disable_jit"]:
        jax.config.update("jax_disable_jit", True)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Create the training function
    train_jit = make_train(config)

    # Run the training
    train_jit(random_key)


if __name__ == "__main__":
    # Set the current working directory to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
