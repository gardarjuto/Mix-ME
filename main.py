import jax
import os
from src.utils.arguments import (
    load_config,
    parse_arguments,
    merge_configs,
    check_config,
)
from src.utils.logging import init_wandb
from src.visualization.visualize import plot_2d_map
from src.training.map_elites import (
    prepare_map_elites_multiagent_hanabi,
    prepare_map_elites_multiagent,
    prepare_map_elites,
    run_training,
    evaluate_adaptation,
)
import wandb


def main():
    args = parse_arguments()

    # Load the configuration file and merge it with the CLI arguments
    config = load_config(args.config)
    config = merge_configs(config, args)

    # Check the configuration
    check_config(config)

    # Initialise WandB
    init_wandb(config)

    config = wandb.config

    if config["disable_jit"]:
        jax.config.update("jax_disable_jit", True)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    if config["multiagent"]:
        if config["env_name"] == "hanabi":
            preparation_fun = prepare_map_elites_multiagent_hanabi
        else:
            preparation_fun = prepare_map_elites_multiagent
    else:
        preparation_fun = prepare_map_elites

    # Init the MAP-Elites algorithm for multi agent
    map_elites, repertoire, emitter_state, random_key = preparation_fun(
        random_key=random_key, **config
    )

    # Run the training
    repertoire, emitter_state, random_key, all_metrics = run_training(
        map_elites, repertoire, emitter_state, random_key=random_key, **config
    )

    # Plot the results
    plot_2d_map(
        repertoire=repertoire,
        **config,
    )

    # Save the repertoire
    if config.get("save_repertoire"):
        repertoire_path = os.path.join(
            config["output_dir"],
            "saved_repertoires",
            wandb.run.name + "-" + wandb.run.id + "/",
        )
        os.makedirs(repertoire_path, exist_ok=True)
        repertoire.save(path=repertoire_path)

    # Evaluate adaptation
    if config.get("adaptation"):
        evaluate_adaptation(
            repertoire=repertoire,
            random_key=random_key,
            **config,
        )


if __name__ == "__main__":
    # Set the current working directory to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
