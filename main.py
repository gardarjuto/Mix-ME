import jax
import os
from src.utils.arguments import load_config, parse_arguments, merge_configs
from src.utils.logging import init_wandb
from src.training.map_elites import (
    run_training,
    prepare_map_elites,
    prepare_map_elites_multiagent,
)


def main():
    args = parse_arguments()

    # Load the configuration file
    config = load_config(args.config)

    # Override the config with the CLI arguments
    config = merge_configs(config, args)

    # Initialise WandB
    init_wandb(config)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    if config["multiagent"]:
        preparation_fun = prepare_map_elites_multiagent
    else:
        preparation_fun = prepare_map_elites

    # Init the MAP-Elites algorithm for multi agent
    map_elites, repertoire, emitter_state, random_key = preparation_fun(
        random_key=random_key, **config
    )

    # Run the training
    run_training(map_elites, repertoire, emitter_state, random_key=random_key, **config)


if __name__ == "__main__":
    # Set the current working directory to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
