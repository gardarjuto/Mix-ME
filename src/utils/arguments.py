import argparse
import yaml

DEFAULT_CONFIG = {
    # Debug
    "disable_jit": False,
    "seed": 42,
    # Logging
    "experiment_name": "DEFAULT",
    "log_period": 10,
    # Environment
    "env_name": "ant_omni",
    "multiagent": False,
    # Hyperparameters
    "batch_size": 16,
    "episode_length": 100,
    "num_iterations": 1000,
    "policy_hidden_layer_sizes": (64, 64),
    "iso_sigma": 0.005,
    "line_sigma": 0.05,
    "num_init_cvt_samples": 50000,
    "num_centroids": 1024,
    "min_bd": 0.0,
    "max_bd": 1.0,
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

    # Logging
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
        "--policy_hidden_layer_sizes",
        type=tuple,
        help="hidden layer sizes of the policy network/s",
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

    return parser.parse_args()


def merge_configs(file_config, cmd_args):
    # Merge dictionaries in the order of priority:
    # command line args > yaml config file > default values
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in file_config.items() if v is not None})
    config.update({k: v for k, v in vars(cmd_args).items() if v is not None})
    return config
