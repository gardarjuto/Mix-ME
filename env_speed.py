import os
import time
import timeit

import jax

from qdax import environments
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from functools import partial

from gymnasium_robotics.envs.multiagent_mujoco import MultiAgentMujocoEnv
from multiprocessing import Process, Pipe

import numpy as np


def make_play_step_fn(
    env,
):
    def play_step_fn(
        env_state,
        random_key,
    ):
        """
        Play a random environment step and return the updated state.
        """
        random_key, subkey = jax.random.split(random_key)
        agent_actions = jax.random.uniform(
            random_key, shape=(env.action_size,), minval=-1.0, maxval=1.0
        )

        next_state = env.step(env_state, agent_actions)

        return next_state, random_key

    return play_step_fn


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state,
    random_key,
    episode_length,
    play_step_fn,
):
    def _scan_play_step_fn(carry, unused_arg):
        env_state, random_key = play_step_fn(*carry)
        return (env_state, random_key), None

    (state, _), _ = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, random_key),
        (),
        length=episode_length,
    )
    return state


def simulate_environments(
    play_step_fn, reset_fn, episode_length, batch_size, random_key
):
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    init_states = reset_fn(keys)
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=random_key,
    )

    _final_state = jax.vmap(unroll_fn)(init_states)
    return _final_state


def simulate_mabrax(config):
    env_names = [
        "ant_uni",
        "halfcheetah_uni",
        "hopper_uni",
        "humanoid_uni",
        "walker2d_uni",
    ]
    random_key = jax.random.PRNGKey(config["seed"])

    for env_name in env_names:
        env = environments.create(env_name, episode_length=config["episode_length"])
        base_env_name = env_name.split("_")[0]
        env = MultiAgentBraxWrapper(
            env,
            env_name=base_env_name,
            parameter_sharing=False,
            emitter_type="naive",
            homogenisation_method="max",
        )
        play_step_fn = jax.jit(make_play_step_fn(env))
        reset_fn = jax.jit(jax.vmap(env.reset))

        # Time the simulation `iterations` times and record the average and std
        times = []
        for _ in range(config["iterations"]):
            start = timeit.default_timer()
            _ = simulate_environments(
                play_step_fn,
                reset_fn,
                config["episode_length"],
                config["batch_size"],
                random_key,
            )
            stop = timeit.default_timer()
            times.append(stop - start)
        print(f"Average time over {config['iterations']} it for {env_name}:")
        print(f"\t{np.mean(times)} +- {np.std(times)}")


def gymnasium_play_step_fn(env):
    """
    Play a random environment step and return the updated state.
    """
    agent_actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, info = env.step(agent_actions)
    return terminations, truncations, info


def simulate_one_environment(env_name, agent_config, episode_length):
    env = MultiAgentMujocoEnv(
        scenario=env_name,
        agent_conf=agent_config,
        max_episode_steps=episode_length,
    )
    env.reset()
    for _ in range(episode_length):
        terminations, truncations, info = gymnasium_play_step_fn(env)
        if truncations["agent_0"] or terminations["agent_0"]:
            env.reset()


def simulate_mamujoco(config):
    env_names = ["Ant", "HalfCheetah", "Hopper", "Humanoid", "Walker2d"]
    agent_configs = {
        "HalfCheetah": "6x1",
        "Ant": "4x2",
        "Hopper": "3x1",
        "Humanoid": "9|8",
        "Walker2d": "2x3",
    }

    for env_name in env_names:
        times = []
        for _ in range(config["iterations"]):
            start = timeit.default_timer()

            # Use multiprocessing to evaluate batch_size environments in parallel
            processes = []
            for _ in range(config["batch_size"]):
                parent_conn, child_conn = Pipe()
                p = Process(
                    target=simulate_one_environment,
                    args=(env_name, agent_configs[env_name], config["episode_length"]),
                )
                p.start()
                processes.append((p, parent_conn, child_conn))

            # Wait for all processes to finish
            for p, parent_conn, child_conn in processes:
                p.join()
                parent_conn.close()
                child_conn.close()

            stop = timeit.default_timer()
            times.append(stop - start)

        print(f"Average time over {config['iterations']} it for {env_name}:")
        print(f"\t{np.mean(times)} +- {np.std(times)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=str, default="mabrax")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.simulator == "mabrax":
        simulate_mabrax(vars(args))
    elif args.simulator == "mamujoco":
        simulate_mamujoco(vars(args))
    else:
        raise NotImplementedError
