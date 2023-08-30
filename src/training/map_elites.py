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
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation
from qdax.core.emitters.ma_standard_emitters import (
    NaiveMultiAgentMixingEmitter,
    MultiAgentEmitter,
)
from qdax.types import (
    EnvState,
    Params,
    RNGKey,
)
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper

from qdax.utils.metrics import default_qd_metrics

from smax.environments.hanabi import HanabiGame

import wandb

from src.utils.generalisation_constants import ADAPTATION_CONSTANTS


def init_multiple_policy_networks(
    env: MultiAgentBraxWrapper,
    policy_hidden_layer_size: int,
) -> dict[int, MLP]:
    action_sizes = env.get_action_sizes()

    policy_networks = {
        agent_idx: MLP(
            layer_sizes=(policy_hidden_layer_size, policy_hidden_layer_size)
            + (action_size,),
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
        for agent_idx, action_size in action_sizes.items()
    }
    return policy_networks


def init_policy_network(
    policy_hidden_layer_size: int,
    action_size: int,
) -> MLP:
    layer_sizes = (policy_hidden_layer_size, policy_hidden_layer_size) + (action_size,)
    policy_network = MLP(
        layer_sizes=layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    return policy_network


def init_controller_population_multiple_networks(
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


def init_controller_population_single_network(
    policy_network: MLP,
    batch_size: int,
    observation_size: int,
    random_key: KeyArray,
):
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    return init_variables


def init_environment_states(
    env: Env | HanabiGame,
    batch_size: int,
    random_key: KeyArray,
):
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    reset_fn = jax.jit(jax.vmap(env.reset))
    if isinstance(env, HanabiGame):
        _, init_states = reset_fn(keys)
    else:
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


def make_policy_network_play_step_fn_hanabi(
    env: HanabiGame,
    policy_network: MLP,
    **kwargs,
) -> Callable[
    [EnvState, Params, RNGKey], tuple[EnvState, Params, RNGKey, QDTransition]
]:
    def play_step_fn(
        env_state,
        policy_params,
        random_key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        # SELECT ACTION
        random_key, _rng = jax.random.split(random_key)

        last_obs = env.get_obs(env_state)
        cur_player = jnp.nonzero(env_state.cur_player_idx, size=1)[0][0]
        legal_moves = env.get_legal_moves(
            env_state.player_hands,
            env_state.fireworks,
            env_state.info_tokens,
            cur_player,
        )[cur_player]
        if not isinstance(policy_params, list):
            policy_params = [policy_params]

        agents_logits = jnp.stack(
            [
                policy_network.apply(params, last_obs[agent])
                for agent, params in zip(env.agents, policy_params)
            ],
            axis=0,
        )

        # Extract logits for the current player
        logits = agents_logits[cur_player]
        logits = jnp.where(legal_moves, logits, -jnp.inf)
        action = jax.random.categorical(_rng, logits, axis=-1)

        # Set same action for all agents (only the current player will be used)
        actions = {agent: action for agent in env.agents}

        obsv, env_state, reward, done, info = env.step(_rng, env_state, actions)

        # Get card_knowledge from env_state
        aidx = jnp.nonzero(env_state.cur_player_idx, size=1)[0][0]
        hand_knowledge = env_state.card_knowledge[aidx]
        # Get knowledge of the played card
        card_knowledge = hand_knowledge[action]
        color_knowledge = card_knowledge[: env.num_colors]
        rank_knowledge = card_knowledge[env.num_colors :]
        fireworks = env_state.fireworks

        def compute_playable_prob(
            color_knowledge: jnp.ndarray,
            rank_knowledge: jnp.ndarray,
            fireworks: jnp.ndarray,
        ):
            """
            Compute the probability that the card is playable

            Args:
                color_knowledge: knowledge of the color of the card. Shape (num_colors,)
                    where each entry is 1 if the card could be that color, 0 if it is
                    known not to be that color.
                rank_knowledge: knowledge of the rank of the card. Shape (num_ranks,)
                    where each entry is 1 if the card could be that rank, 0 if it is
                    known not to be that rank.
                fireworks: current state of the fireworks. Shape (num_colors, num_ranks)
                    where the sub-array for each color is the thermometer encoded rank
                    of the last played card of that color. The first zero entry is the
                    rank of the next playable card of that color.

            Returns:
                probability that the card is playable
            """
            color_knowledge_exp = jnp.expand_dims(color_knowledge, axis=-1).astype(
                jnp.int32
            )
            rank_knowledge_exp = jnp.expand_dims(rank_knowledge, axis=0).astype(
                jnp.int32
            )

            # Compute outer product of color and rank knowledge
            knowledge_outer = color_knowledge_exp * rank_knowledge_exp

            # Get the rank of the next playable card for each color in the fireworks
            next_playable_ranks = jnp.sum(fireworks, axis=-1).astype(jnp.int32)

            # Create a mask for which colors are not fully played
            full_mask = jnp.expand_dims(next_playable_ranks == 5, axis=-1).squeeze()

            # Get the knowledge for each color and the next playable rank, but only for
            # the playable colors
            next_playable_knowledge = jnp.where(
                ~full_mask,
                jnp.take_along_axis(
                    knowledge_outer,
                    jnp.expand_dims(next_playable_ranks, axis=-1),
                    axis=-1,
                ).squeeze(),
                0,
            )

            # Sum the knowledge and normalize by the sum of color and rank knowledge to
            # get the probability
            prob = jnp.sum(next_playable_knowledge) / jnp.sum(knowledge_outer)

            return prob

        is_hint = (2 * env.hand_size) <= action
        is_play = (env.hand_size <= action) & (action < (2 * env.hand_size))
        hint_available = jnp.sum(env_state.info_tokens) > 0

        hint_desc = jnp.where(
            is_hint & hint_available, 1, jnp.where(hint_available, 0, -1)
        )
        playable_prob = jnp.where(
            is_play,
            compute_playable_prob(color_knowledge, rank_knowledge, fireworks),
            -1,
        )

        # Compute the descriptor
        desc = jnp.array(
            [
                playable_prob,
                hint_desc,
            ]
        )

        next_desc = None

        transition = QDTransition(
            obs=last_obs,
            next_obs=obsv,
            rewards=reward,
            dones=done,
            actions=action,
            truncations=None,
            state_desc=desc,
            next_state_desc=next_desc,
        )

        return env_state, policy_params, random_key, transition

    return play_step_fn


def scoring_function_hanabi(
    policies_params,
    random_key,
    play_step_fn,
    behavior_descriptor_extractor,
    batch_size,
    sample_size,
    episode_length,
    env_reset_fn,
):
    # Perform rollouts with each policy
    unroll_fn = functools.partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )

    def one_est(policies_params, random_key):
        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=batch_size)
        reset_fn = jax.jit(jax.vmap(env_reset_fn))
        _, init_states = reset_fn(keys)

        keys = jax.random.split(random_key, batch_size)
        _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params, keys)

        # create a mask to extract data properly
        is_done = jnp.clip(jnp.cumsum(data.dones["__all__"], axis=1), 0, 1)
        mask = jnp.roll(is_done, 1, axis=1)
        mask = mask.at[:, 0].set(0)

        # scores
        fitnesses = jnp.sum(data.rewards["__all__"] * (1.0 - mask), axis=1)
        descriptors = behavior_descriptor_extractor(data, mask)

        return fitnesses, descriptors

    keys = jax.random.split(random_key, sample_size)
    fitnesses, descriptors = jax.vmap(one_est, in_axes=(None, 0))(policies_params, keys)
    # Mean over samples
    fitnesses = jnp.mean(fitnesses, axis=0)
    descriptors = jnp.mean(descriptors, axis=0)

    return (
        fitnesses,
        descriptors,
        None,
        random_key,
    )


def prepare_map_elites_multiagent_hanabi(
    env_name: str,
    batch_size: int,
    sample_size: int,
    episode_length: int,
    policy_hidden_layer_size: int,
    parameter_sharing: bool,
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    k_mutations: int,
    emitter_type: str,
    homogenisation_method: str,
    eta: float,
    mut_val_bound: float,
    proportion_to_mutate: float,
    variation_percentage: float,
    crossplay_percentage: float,
    random_key: KeyArray,
    **kwargs,
):
    # Create environment
    num_agents = 2
    env = HanabiGame(num_agents=num_agents)

    # Init policy network
    policy_network = init_policy_network(
        policy_hidden_layer_size, env.action_space(env.agents[0]).n
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_agents * batch_size)

    init_variables = []
    for agent_keys in jnp.split(keys, num_agents, axis=0):
        fake_batch = jnp.zeros(shape=(batch_size, env.observation_space("agent_0").n))
        init_variables.append(jax.vmap(policy_network.init)(agent_keys, fake_batch))

    # Create the play step function
    play_step_fn = make_policy_network_play_step_fn_hanabi(env, policy_network)

    # Prepare the scoring function
    def bd_extraction_fn(data, mask):
        playable_prob = data.state_desc[:, :, 0]
        hint_desc = data.state_desc[:, :, 1]
        average_playable_prob = jnp.sum(
            jnp.where(playable_prob < 0, 0, playable_prob), axis=1
        ) / jnp.sum(playable_prob != -1, axis=1)
        communicativeness = jnp.sum(hint_desc == 1, axis=1) / jnp.sum(
            hint_desc != -1, axis=1
        )
        return jnp.stack([average_playable_prob, communicativeness], axis=1)

    scoring_fn = functools.partial(
        scoring_function_hanabi,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
        batch_size=batch_size,
        sample_size=sample_size,
        episode_length=episode_length,
        env_reset_fn=env.reset,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )
    mutation_function = functools.partial(
        polynomial_mutation,
        eta=eta,
        minval=-mut_val_bound,
        maxval=mut_val_bound,
        proportion_to_mutate=proportion_to_mutate,
    )

    if emitter_type == "naive":
        emitter = NaiveMultiAgentMixingEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
            num_agents=num_agents,
            agents_to_mutate=k_mutations,
        )
    else:
        emitter = MultiAgentEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            crossplay_percentage=crossplay_percentage,
            batch_size=batch_size,
            num_agents=num_agents,
            role_preserving=emitter_type == "role_preserving",
            agents_to_mutate=k_mutations,
        )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_function,
        qd_offset=reward_offset,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=2,
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


def prepare_map_elites_multiagent(
    env_name: str,
    batch_size: int,
    episode_length: int,
    policy_hidden_layer_size: int,
    parameter_sharing: bool,
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    k_mutations: int,
    emitter_type: str,
    homogenisation_method: str,
    eta: float,
    mut_val_bound: float,
    proportion_to_mutate: float,
    variation_percentage: float,
    crossplay_percentage: float,
    random_key: KeyArray,
    **kwargs,
):
    # Create environment
    base_env_name = env_name.split("_")[0]
    env = environments.create(env_name, episode_length=episode_length)
    env = MultiAgentBraxWrapper(
        env,
        env_name=base_env_name,
        parameter_sharing=parameter_sharing,
        emitter_type=emitter_type,
        homogenisation_method=homogenisation_method,
    )
    num_agents = len(env.get_action_sizes())

    # Init policy network/s
    if parameter_sharing:
        if homogenisation_method == "concat":
            policy_network = init_policy_network(
                policy_hidden_layer_size, env.action_size
            )
        else:
            policy_network = init_policy_network(
                policy_hidden_layer_size, env.get_action_sizes()[0]
            )
        init_variables = init_controller_population_multiple_networks(
            env, {0: policy_network}, batch_size, random_key
        )[0]
    else:
        policy_network = {
            agent_idx: init_policy_network(policy_hidden_layer_size, action_size)
            for agent_idx, action_size in env.get_action_sizes().items()
        }
        init_variables = init_controller_population_multiple_networks(
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
    mutation_function = functools.partial(
        polynomial_mutation,
        eta=eta,
        minval=-mut_val_bound,
        maxval=mut_val_bound,
        proportion_to_mutate=proportion_to_mutate,
    )

    if parameter_sharing:
        emitter = MixingEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
        )
    elif emitter_type == "naive":
        emitter = NaiveMultiAgentMixingEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
            num_agents=num_agents,
            agents_to_mutate=k_mutations,
        )
    else:
        emitter = MultiAgentEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            crossplay_percentage=crossplay_percentage,
            batch_size=batch_size,
            num_agents=num_agents,
            role_preserving=emitter_type == "role_preserving",
            agents_to_mutate=k_mutations,
        )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_function,
        qd_offset=reward_offset,
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
    sample_size: int,
    episode_length: int,
    policy_hidden_layer_size: int,
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    eta: float,
    mut_val_bound: float,
    proportion_to_mutate: float,
    variation_percentage: float,
    random_key: KeyArray,
    **kwargs,
):
    # Create environment
    if env_name == "hanabi":
        env = HanabiGame()
    else:
        env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    action_size = (
        env.action_size if env_name != "hanabi" else env.action_space(env.agents[0]).n
    )
    policy_network = init_policy_network(policy_hidden_layer_size, action_size)

    # Init population of controllers
    observation_size = (
        env.observation_size
        if env_name != "hanabi"
        else env.observation_space(env.agents[0]).n
    )
    init_variables = init_controller_population_single_network(
        policy_network, batch_size, observation_size, random_key
    )

    # Create the initial environment states
    init_states = init_environment_states(env, batch_size, random_key)

    # Create the play step function
    if env_name == "hanabi":
        play_step_fn = make_policy_network_play_step_fn_hanabi(env, policy_network)
    else:
        play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)

    # Prepare the scoring function
    if env_name == "hanabi":

        def bd_extraction_fn(data, mask):
            playable_prob = data.state_desc[:, :, 0]
            hint_desc = data.state_desc[:, :, 1]
            average_playable_prob = jnp.sum(
                jnp.where(playable_prob < 0, 0, playable_prob), axis=1
            ) / jnp.sum(playable_prob != -1, axis=1)
            communicativeness = jnp.sum(hint_desc == 1, axis=1) / jnp.sum(
                hint_desc != -1, axis=1
            )
            return jnp.stack([average_playable_prob, communicativeness], axis=1)

        scoring_fn = functools.partial(
            scoring_function_hanabi,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
            batch_size=batch_size,
            sample_size=sample_size,
            episode_length=episode_length,
            env_reset_fn=env.reset,
        )
    else:
        bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
        scoring_fn = functools.partial(
            scoring_function,
            init_states=init_states,
            episode_length=episode_length,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name] if env_name != "hanabi" else 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )
    mutation_function = functools.partial(
        polynomial_mutation,
        eta=eta,
        minval=-mut_val_bound,
        maxval=mut_val_bound,
        proportion_to_mutate=proportion_to_mutate,
    )

    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_function,
        variation_fn=variation_fn,
        variation_percentage=variation_percentage,
        batch_size=batch_size,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        qd_offset=reward_offset,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length if env_name != "hanabi" else 2,
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


def evaluate_adaptation(
    repertoire: MapElitesRepertoire,
    adaptation_eval_num: int,
    env_name: str,
    episode_length: int,
    policy_hidden_layer_size: int,
    parameter_sharing: bool,
    emitter_type: str,
    homogenisation_method: str,
    multiagent: bool,
    random_key: KeyArray,
    **kwargs,
):
    base_env_name = env_name.split("_")[0]

    fitnesses = []

    for adaptation_name in ADAPTATION_CONSTANTS:
        adaptation_constants = ADAPTATION_CONSTANTS[adaptation_name]
        adaptation_constants_env = adaptation_constants[env_name]

        for adaptation_idx in range(10):
            env_kwargs = {}
            env_kwargs[adaptation_name] = jax.tree_map(
                lambda x: x[adaptation_idx], adaptation_constants_env
            )

            eval_env = environments.create(
                env_name=env_name,
                batch_size=None,
                episode_length=episode_length,
                auto_reset=True,
                eval_metrics=True,
                **env_kwargs,
            )
            if multiagent:
                eval_env = MultiAgentBraxWrapper(
                    eval_env,
                    env_name=base_env_name,
                    parameter_sharing=parameter_sharing,
                    emitter_type=emitter_type,
                    homogenisation_method=homogenisation_method,
                )
                policy_network = init_multiple_policy_networks(
                    eval_env, policy_hidden_layer_size
                )
                play_step_fn = make_policy_network_play_step_fn(
                    eval_env, policy_network, parameter_sharing
                )
            else:
                policy_network = init_policy_network(
                    policy_hidden_layer_size, eval_env.action_size
                )
                play_step_fn = make_policy_network_play_step_fn_brax(
                    eval_env, policy_network
                )

            bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
            scoring_fn = functools.partial(
                scoring_function,
                episode_length=episode_length,
                play_step_fn=play_step_fn,
                behavior_descriptor_extractor=bd_extraction_fn,
            )

            scoring_fn = jax.jit(scoring_fn)
            reset_fn = jax.jit(jax.vmap(eval_env.reset))

            # Extract the policies
            policies = jax.tree_map(
                lambda x: x[repertoire.fitnesses != -jnp.inf], repertoire.genotypes
            )
            num_policies = jax.tree_util.tree_leaves(policies)[0].shape[0]

            # Define a helper function to evaluate policies
            @jax.jit
            def evaluate_policies_helper(random_key):
                keys = jax.random.split(random_key, num=num_policies)
                init_states = reset_fn(keys)
                eval_fitnesses, descriptors, extra_scores, random_key = scoring_fn(
                    policies, random_key, init_states
                )
                return eval_fitnesses

            # Generate random keys for each evaluation
            random_key, subkey = jax.random.split(random_key)
            keys = jax.random.split(subkey, num=adaptation_eval_num)

            # Parallelize the evaluation
            eval_fitnesses = jax.vmap(evaluate_policies_helper)(keys)

            # Compute the median fitness for each policy over its states
            median_fitnesses = jnp.median(eval_fitnesses, axis=0)

            # Report the highest median fitness
            fitnesses.append(
                (adaptation_name, adaptation_idx, jnp.max(median_fitnesses).item())
            )

    table = wandb.Table(
        columns=["adaptation_name", "adaptation_idx", "adaptation_fitness"],
        data=fitnesses,
    )
    wandb.log({"adaptation_fitness": table})
