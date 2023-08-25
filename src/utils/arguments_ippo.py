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
    # Environment
    "env_name": "halfcheetah_uni",
    # Config for IPPO
    "lr": 2.5e-4,
    "num_envs": 32,
    "num_steps": 300,
    "total_timesteps": 7e6,
    "update_epochs": 4,
    "num_minibatches": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "activation": "tanh",
    "anneal_lr": True,
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
        action="store_true",
        help="whether to disable JIT compilation",
    )
    parser.add_argument(
        "--no-disable_jit",
        dest="disable_jit",
        action="store_false",
    )
    parser.set_defaults(disable_jit=False)
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

    # Environment
    parser.add_argument(
        "--env_name",
        type=str,
        help="name of the environment to use",
    )

    # Config for IPPO
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        help="number of environments",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        help="length of each rollout",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        help="total number of timesteps",
    )
    parser.add_argument(
        "--update_epochs",
        type=int,
        help="number of epochs to update the policy",
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        help="number of minibatches to update the policy",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="discount factor",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        help="lambda parameter for GAE",
    )
    parser.add_argument(
        "--clip_eps",
        type=float,
        help="epsilon parameter for PPO clipping",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        help="coefficient for the entropy loss",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        help="coefficient for the value function loss",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help="maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="activation function to use",
    )
    parser.add_argument(
        "--anneal_lr",
        action="store_true",
        help="whether to anneal the learning rate",
    )
    parser.add_argument(
        "--no-anneal_lr",
        dest="anneal_lr",
        action="store_false",
    )
    parser.set_defaults(anneal_lr=True)

    return parser.parse_args()


def merge_configs(file_config, cmd_args):
    # Merge dictionaries in the order of priority:
    # command line args > yaml config file > default values
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in file_config.items() if v is not None})
    config.update({k: v for k, v in vars(cmd_args).items() if v is not None})
    return config
