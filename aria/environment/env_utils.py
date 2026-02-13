import os
import json
import time
import random
from typing import List, Dict, Any, Tuple
import sys
import torch
import numpy as np
import wandb
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()
import pdb
from accelerate import Accelerator
from twenty_questions import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv
from guess_my_city import GuessMyCityEnv, BatchedGuessMyCityEnv
from bargaining import BargainingEnv, BARGAINING_CONFIGS
from negotiation import NegotiationEnv, NEGOTIATION_CONFIGS
from llm_base import llm_openai, vllm


def collect_environment_data(env_name: str, agent, word_idx: int = None,
                             config_idx: int = None,
                             agent_role: str = "alice",
                             opponent_model: str = "gpt-4o") -> Tuple[List[Dict], Dict]:
    """
    Collect trajectory data from a specific environment with an agent.

    Args:
        env_name:       Name of the environment to use.
                        One of: 'twenty_questions', 'guess_my_city',
                                'bargaining', 'negotiation'
        agent:          Agent to interact with the environment.
        word_idx:       Optional index for single-agent envs (which word/city).
        config_idx:     Optional config index for adversarial envs (0-47).
        agent_role:     'alice' or 'bob' – only used for adversarial envs.
        opponent_model: LLM model name for the opponent – only used for
                        adversarial envs.

    Returns:
        Tuple containing:
          - list of trajectory dictionaries with state/action/reward info
          - metadata dictionary with summary statistics
    """
    # ── Single-agent environments ──────────────────────────────────────────────
    if env_name == "twenty_questions":
        env = TwentyQuestionsEnv()
        word_list_key = "curr_word[0]"

        init = env.reset(idx=word_idx) if word_idx is not None else env.reset()
        next_obs = init
        steps, done = 0, False
        trajectory = []

        while not done:
            steps += 1
            observation = next_obs[0]["content"]
            action = agent.get_action([observation])[0]
            next_obs, answer, reward, done = env._step(question=action)
            curr_word = env.curr_word[0]

            trajectory.append({
                "instruction": init[0]["content"],
                "curr_world": curr_word,
                "question": action,
                "answer": answer,
                "observation": observation,
                "action": action,
                "next_observation": next_obs[0]["content"],
                "reward": reward,
                "done": done,
            })
            if steps > env.max_conversation_length:
                break

    elif env_name == "guess_my_city":
        env = GuessMyCityEnv()

        init = env.reset(idx=word_idx) if word_idx is not None else env.reset()
        next_obs = init
        steps, done = 0, False
        trajectory = []

        while not done:
            steps += 1
            observation = next_obs[0]["content"]
            action = agent.get_action([observation])[0]
            next_obs, answer, reward, done = env._step(question=action)
            curr_word = env.curr_word

            trajectory.append({
                "instruction": init[0]["content"],
                "curr_world": curr_word,
                "question": action,
                "answer": answer,
                "observation": observation,
                "action": action,
                "next_observation": next_obs[0]["content"],
                "reward": reward,
                "done": done,
            })
            if steps > env.max_conversation_length:
                break

    # ── Adversarial environments ───────────────────────────────────────────────
    elif env_name == "bargaining":
        env = BargainingEnv(opponent_model=opponent_model, agent_role=agent_role)
        init = env.reset(config_idx=config_idx)
        next_obs = init
        steps, done = 0, False
        trajectory = []

        while not done:
            steps += 1
            observation = next_obs[0]["content"]
            action = agent.get_action([observation])[0]
            next_obs, opponent_response, reward, done = env._step(question=action)

            trajectory.append({
                "instruction": init[0]["content"],
                "game_config": env.get_game_data()["config"],
                "agent_role": agent_role,
                "observation": observation,
                "action": action,
                "opponent_response": opponent_response,
                "next_observation": next_obs[0]["content"],
                "reward": reward,
                "done": done,
            })
            if steps > env.max_rounds * 2 + 2:
                break

        # Attach full-game data (all actions by both players) to every step
        game_data = env.get_game_data()
        for item in trajectory:
            item["alice_actions"] = game_data["alice_actions"]
            item["bob_actions"] = game_data["bob_actions"]
            item["alice_ids"] = game_data["alice_ids"]
            item["bob_ids"] = game_data["bob_ids"]
            item["game_result"] = game_data["game_result"]

    elif env_name == "negotiation":
        env = NegotiationEnv(opponent_model=opponent_model, agent_role=agent_role)
        init = env.reset(config_idx=config_idx)
        next_obs = init
        steps, done = 0, False
        trajectory = []

        while not done:
            steps += 1
            observation = next_obs[0]["content"]
            action = agent.get_action([observation])[0]
            next_obs, opponent_response, reward, done = env._step(question=action)

            trajectory.append({
                "instruction": init[0]["content"],
                "game_config": env.get_game_data()["config"],
                "agent_role": agent_role,
                "observation": observation,
                "action": action,
                "opponent_response": opponent_response,
                "next_observation": next_obs[0]["content"],
                "reward": reward,
                "done": done,
            })
            if steps > env.max_rounds * 2 + 2:
                break

        game_data = env.get_game_data()
        for item in trajectory:
            item["alice_actions"] = game_data["alice_actions"]
            item["bob_actions"] = game_data["bob_actions"]
            item["alice_ids"] = game_data["alice_ids"]
            item["bob_ids"] = game_data["bob_ids"]
            item["game_result"] = game_data["game_result"]

    else:
        raise NotImplementedError(f"Environment '{env_name}' not implemented")

    # ── Common trajectory post-processing ─────────────────────────────────────
    total_reward = sum(item["reward"] for item in trajectory)
    final_reward = trajectory[-1]["reward"] if trajectory else 0
    discount_factor = 0.95
    discount_return = sum(
        pow(discount_factor, len(trajectory) - 1 - idx) * final_reward
        for idx in range(len(trajectory))
    )
    for item in trajectory:
        item.update({"trajectory_reward": total_reward, "mc_return": discount_return})

    # ── Build metadata ─────────────────────────────────────────────────────────
    if env_name in ("twenty_questions", "guess_my_city"):
        curr_word = trajectory[-1].get("curr_world", "") if trajectory else ""
        metadata = {
            "total_reward": total_reward,
            "final_reward": final_reward,
            "length": len(trajectory),
            "curr_word": curr_word,
            "success": done and final_reward == 0,
        }
    else:
        game_result = trajectory[-1].get("game_result") if trajectory else None
        metadata = {
            "total_reward": total_reward,
            "final_reward": final_reward,
            "length": len(trajectory),
            "game_result": game_result,
            "env_name": env_name,
            "agent_role": agent_role,
            "config_idx": config_idx,
        }

    return trajectory, metadata


def interact_with_environment(env_name: str, agent, output_dir: str,
                              repeat: int,
                              opponent_model: str = "gpt-4o") -> List[Dict]:
    """
    Run multiple interactions with an environment and save trajectories.

    Args:
        env_name:       Name of the environment to use.
        agent:          Agent to interact with the environment.
        output_dir:     Directory to save trajectory data.
        repeat:         Number of trajectories to generate (per config for
                        adversarial envs).
        opponent_model: LLM model for the opponent (adversarial envs only).

    Returns:
        List of all trajectory dictionaries collected.
    """
    env_output_dir = f"{output_dir}/{env_name}"
    os.makedirs(env_output_dir, exist_ok=True)
    all_trajectories = []

    if env_name in ("twenty_questions", "guess_my_city"):
        for variation in tqdm(range(repeat), desc=f"Generating {env_name} trajectories"):
            variation_path = f"{env_output_dir}/variation-{variation}.jsonl"
            trajectory, _ = collect_environment_data(env_name, agent)
            if trajectory:
                with open(variation_path, "w") as f:
                    for item in trajectory:
                        f.write(json.dumps(item) + "\n")
                all_trajectories.extend(trajectory)

    elif env_name in ("bargaining", "negotiation"):
        configs = BARGAINING_CONFIGS if env_name == "bargaining" else NEGOTIATION_CONFIGS
        for role in ("alice", "bob"):
            for cfg_idx in tqdm(range(len(configs)),
                                desc=f"Generating {env_name}/{role} trajectories"):
                for rep in range(repeat):
                    variation_path = (
                        f"{env_output_dir}/{role}_config{cfg_idx}_rep{rep}.jsonl"
                    )
                    trajectory, _ = collect_environment_data(
                        env_name, agent,
                        config_idx=cfg_idx,
                        agent_role=role,
                        opponent_model=opponent_model,
                    )
                    if trajectory:
                        with open(variation_path, "w") as f:
                            for item in trajectory:
                                f.write(json.dumps(item) + "\n")
                        all_trajectories.extend(trajectory)
    else:
        raise NotImplementedError(f"Environment '{env_name}' not implemented")

    return all_trajectories


def get_word_list(env_name: str):
    """
    Get the list of possible words/targets for a specific environment.

    Args:
        env_name: Name of the environment.

    Returns:
        List of words/targets, or list of config tuples for adversarial envs.
    """
    if env_name == "twenty_questions":
        from twenty_questions import DEFAULT_OBJECT_LIST
        word_list = DEFAULT_OBJECT_LIST
        return [list(map(lambda x: x.lower(), word.split(";"))) for word in word_list]
    elif env_name == "guess_my_city":
        from guess_my_city import CITY_LIST
        return CITY_LIST
    elif env_name == "bargaining":
        return BARGAINING_CONFIGS
    elif env_name == "negotiation":
        return NEGOTIATION_CONFIGS
    raise ValueError(f"Unknown environment: {env_name}")


def batch_interact_environment(agent, iteration: int, sample_size: int = None,
                               n_rollout: int = None, agent_type: str = "",
                               env_names: List[str] = None,
                               opponent_model: str = "gpt-4o") -> List[Dict]:
    """
    Run agent interactions with multiple environments and collect trajectory data.

    Args:
        agent:          Agent providing actions.
        iteration:      Current iteration number (used for output directory naming).
        sample_size:    Number of samples to collect per environment.
        n_rollout:      Not used; kept for compatibility.
        agent_type:     Agent type identifier for output directory.
        env_names:      List of environment names to use. Defaults to all four.
        opponent_model: LLM model for adversarial opponents.

    Returns:
        List of all trajectory data from all environments.
    """
    if env_names is None:
        env_names = ["twenty_questions", "guess_my_city", "bargaining", "negotiation"]

    if agent_type:
        output_dir = f"outputs/{agent_type}/trajectories/iter_{iteration}/"
    else:
        output_dir = f"outputs/trajectories/iter_{iteration}/"

    all_outputs = []
    for env_name in env_names:
        outputs = interact_with_environment(
            env_name, agent, output_dir,
            sample_size or 10,
            opponent_model=opponent_model,
        )
        all_outputs.extend(outputs)

    return all_outputs
