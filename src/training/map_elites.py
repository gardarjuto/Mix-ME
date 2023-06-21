import functools
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from brax.envs import Env
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax import environments
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    scoring_function_brax_envs as scoring_function,
)
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.ma_standard_emitters import MultiAgentMixingEmitter
from qdax.types import (
    EnvState,
    Params,
    RNGKey,
)
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper

from qdax.utils.metrics import default_qd_metrics

import wandb


def init_multiple_policy_networks(
    env: MultiAgentBraxWrapper,
    policy_hidden_layer_sizes: list[int],
) -> dict[int, MLP]:
    action_sizes = env.get_action_sizes()

    policy_networks = {
        agent_idx: MLP(
            layer_sizes=tuple(policy_hidden_layer_sizes) + (action_size,),
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
        for agent_idx, action_size in action_sizes.items()
    }
    return policy_networks


def init_policy_network(
    policy_hidden_layer_sizes: list[int],
    action_size: int,
) -> MLP:
    layer_sizes = tuple(policy_hidden_layer_sizes) + (action_size,)
    policy_network = MLP(
        layer_sizes=layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    return policy_network


def init_controller_population_multiagent(
    env: MultiAgentBraxWrapper,
    policy_networks: dict[int, MLP],
    batch_size: int,
    random_key: KeyArray,
):
    num_agents = len(policy_networks)
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_agents * batch_size)

    init_variables = []
    for (agent_idx, agent_policy), agent_keys in zip(
        policy_networks.items(), jnp.split(keys, num_agents, axis=0)
    ):
        fake_batch = jnp.zeros(shape=(batch_size, env.get_obs_sizes()[agent_idx]))
        init_variables.append(jax.vmap(agent_policy.init)(agent_keys, fake_batch))

    return init_variables


def init_controller_population_single_agent(
    env: Env,
    policy_network: MLP,
    batch_size: int,
    random_key: KeyArray,
):
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    return init_variables


def init_environment_states(
    env: Env,
    batch_size: int,
    random_key: KeyArray,
):
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)
    return init_states


def make_policy_network_play_step_fn(
    env: MultiAgentBraxWrapper,
    policy_network: dict[int, MLP] | MLP,
    parameter_sharing: bool,
) -> Callable[
    [EnvState, Params, RNGKey], tuple[EnvState, Params, RNGKey, QDTransition]
]:
    def play_step_fn(
        env_state: EnvState,
        policy_params: list[Params] | Params,
        random_key: KeyArray,
    ) -> tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated state and the transition.
        """
        obs = env.obs(env_state)
        if not parameter_sharing:
            agent_actions = {
                agent_idx: network.apply(params, agent_obs)
                for (agent_idx, network), params, agent_obs in zip(
                    policy_network.items(), policy_params, obs.values()
                )
            }
        else:
            agent_actions = {
                agent_idx: policy_network.apply(policy_params, agent_obs)
                for (agent_idx, agent_obs) in obs.items()
            }

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, agent_actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=agent_actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    return play_step_fn


def prepare_map_elites_multiagent(
    env_name: str,
    batch_size: int,
    episode_length: int,
    policy_hidden_layer_sizes: list[int],
    parameter_sharing: bool,
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    k_mutations: int,
    random_key: KeyArray,
    **kwargs,
):
    # Create environment
    base_env_name = env_name.split("_")[0]
    env = environments.create(env_name, episode_length=episode_length)
    env = MultiAgentBraxWrapper(
        env, env_name=base_env_name, parameter_sharing=parameter_sharing
    )
    num_agents = len(env.get_action_sizes())

    # Init policy network/s
    if parameter_sharing:
        policy_network = init_policy_network(policy_hidden_layer_sizes, env.action_size)
        init_variables = init_controller_population_single_agent(
            env, policy_network, batch_size, random_key
        )
    else:
        policy_network = {
            agent_idx: init_policy_network(policy_hidden_layer_sizes, action_size)
            for agent_idx, action_size in env.get_action_sizes().items()
        }
        init_variables = init_controller_population_multiagent(
            env, policy_network, batch_size, random_key
        )

    # Create the initial environment states
    init_states = init_environment_states(env, batch_size, random_key)

    # Create the play step function
    play_step_fn = make_policy_network_play_step_fn(
        env, policy_network, parameter_sharing
    )

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )

    if parameter_sharing:
        mixing_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
        )
    else:
        mixing_emitter = MultiAgentMixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
            num_agents=num_agents,
            agents_to_mutate=k_mutations,
        )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    return map_elites, repertoire, emitter_state, random_key


def prepare_map_elites(
    env_name: str,
    batch_size: int,
    episode_length: int,
    policy_hidden_layer_sizes: list[int],
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    random_key: KeyArray,
    **kwargs,
):
    # Create environment
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_network = init_policy_network(policy_hidden_layer_sizes, env.action_size)

    # Init population of controllers
    init_variables = init_controller_population_single_agent(
        env, policy_network, batch_size, random_key
    )

    # Create the initial environment states
    init_states = init_environment_states(env, batch_size, random_key)

    # Create the play step function
    play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    return map_elites, repertoire, emitter_state, random_key


def run_training(
    map_elites: MAPElites,
    repertoire: MapElitesRepertoire,
    emitter_state: EmitterState,
    num_iterations: int,
    log_period: int,
    random_key: KeyArray,
    **kwargs,
):
    num_loops = int(num_iterations / log_period)

    # Prepare the logger
    all_metrics = {}

    # main loop
    map_elites_scan_update = map_elites.scan_update
    for i in range(num_loops):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            random_key,
        ), metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )

        timelapse = time.time() - start_time

        # log metrics
        logged_metrics = {
            "time": timelapse,
        }
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        wandb.log(logged_metrics, step=1 + i * log_period)

    return repertoire, emitter_state, random_key, all_metrics
