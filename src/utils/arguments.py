import argparse
import yaml

DEFAULT_CONFIG = {
    # Debug
    "disable_jit": False,
    "seed": 42,
    "output_dir": "output",
    # Logging
    "project_name": "MA-QD",
    "entity_name": "ucl-dark",
    "experiment_name": None,
    "log_period": 10,
    "wandb_mode": "offline",
    "save_repertoire": False,
    # Environment
    "env_name": "halfcheetah_uni",
    "multiagent": False,
    # Hyperparameters
    "batch_size": 16,
    "sample_size": 20,
    "episode_length": 100,
    "num_iterations": 1000,
    "policy_hidden_layer_size": 64,
    "parameter_sharing": False,
    "iso_sigma": 0.005,
    "line_sigma": 0.05,
    "num_init_cvt_samples": 50000,
    "num_centroids": 1024,
    "min_bd": 0.0,
    "max_bd": 1.0,
    "k_mutations": 1,
    "emitter_type": "naive",
    "homogenisation_method": "concat",
    "eta": 10.0,
    "mut_val_bound": 0.5,
    "proportion_to_mutate": 0.1,
    "variation_percentage": 0.3,
    "crossplay_percentage": 0.3,
    # Adaptation
    "adaptation": False,
    "adaptation_eval_num": 100,
}


def load_config(file_path):
    # Load a yaml configuration file
    with open(file_path, "r") as fd:
        try:
            return yaml.load(fd, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Quality Diversity Training Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="path to the configuration file",
    )

    # Debug
    parser.add_argument(
        "--disable_jit",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="whether to disable JIT compilation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed for the random number generators",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to the output directory",
    )

    # Logging
    parser.add_argument(
        "--project_name",
        type=str,
        help="name of the project",
    )
    parser.add_argument(
        "--entity_name",
        type=str,
        help="name of the wandb entity",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="name of the experiment",
    )
    parser.add_argument(
        "--log_period",
        type=int,
        help="number of iterations between each logging",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        help="mode of WandB",
    )
    parser.add_argument(
        "--save_repertoire",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="whether to save the final MAP-Elites repertoire",
    )

    # Environment
    parser.add_argument(
        "--env_name",
        type=str,
        help="name of the environment to use",
    )
    parser.add_argument(
        "--multiagent",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="whether to use the multi-agent version of MAP-Elites",
    )

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="number of samples to average over in noisy evaluation",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        help="length of each episode",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="number of iterations to run",
    )
    parser.add_argument(
        "--policy_hidden_layer_size",
        type=int,
        help="size of the hidden layers of the policy network/s",
    )
    parser.add_argument(
        "--parameter_sharing",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="whether to use parameter sharing between agents",
    )
    parser.add_argument(
        "--iso_sigma",
        type=float,
        help="isotropic sigma",
    )
    parser.add_argument(
        "--line_sigma",
        type=float,
        help="line sigma",
    )
    parser.add_argument(
        "--num_init_cvt_samples",
        type=int,
        help="number of initial samples for CVT",
    )
    parser.add_argument(
        "--num_centroids",
        type=int,
        help="number of centroids for CVT",
    )
    parser.add_argument(
        "--min_bd",
        type=float,
        help="minimum bound of the behavior descriptor",
    )
    parser.add_argument(
        "--max_bd",
        type=float,
        help="maximum bound of the behavior descriptor",
    )
    parser.add_argument(
        "--emitter_type",
        type=str,
        help="type of emitter to use",
    )
    parser.add_argument(
        "--k_mutations",
        type=int,
        help="number of agents to mutate each iteration",
    )
    parser.add_argument(
        "--homogenisation_method",
        type=str,
        help="method to use for joining dimensions of heterogeneous agents (concat, max)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="eta parameter for polynomial mutation",
    )
    parser.add_argument(
        "--mut_val_bound",
        type=float,
        help="bound for polynomial mutation (-mut_val_bound, mut_val_bound)",
    )
    parser.add_argument(
        "--proportion_to_mutate",
        type=float,
        help="proportion of the population to mutate",
    )
    parser.add_argument(
        "--variation_percentage",
        type=float,
        help="percentage of the population to mutate",
    )
    parser.add_argument(
        "--crossplay_percentage",
        type=float,
        help="percentage of the population to crossplay",
    )
    # Adaptation
    parser.add_argument(
        "--adaptation",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="whether to run adaptation",
    )
    parser.add_argument(
        "--adaptation_eval_num",
        type=int,
        help="number of evaluations to run for adaptation",
    )

    return parser.parse_args()


def merge_configs(file_config, cmd_args):
    # Merge dictionaries in the order of priority:
    # command line args > yaml config file > default values
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in file_config.items() if v is not None})
    config.update({k: v for k, v in vars(cmd_args).items() if v is not None})
    return config


def check_config(config):
    # Check that the configuration is valid
    assert config["batch_size"] > 0
    assert config["episode_length"] > 0
    assert config["num_iterations"] > 0
    assert config["policy_hidden_layer_size"] > 0
    assert config["iso_sigma"] > 0.0
    assert config["line_sigma"] > 0.0
    assert config["num_init_cvt_samples"] > 0
    assert config["num_centroids"] > 0
    assert config["min_bd"] < config["max_bd"]
    assert config["homogenisation_method"] in ["concat", "max"]
    return True
