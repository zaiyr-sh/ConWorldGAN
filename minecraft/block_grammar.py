from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import torch

def get_sentence(block_name, is_block_plural, world_name, is_world_plural, unmasker):
    if is_block_plural:
        sentence_start = f"These {block_name} are "
    else:
        sentence_start = f"This {block_name} is "

    if is_world_plural:
        sentence_end = f" these {world_name}."
    else:
        sentence_end = f" this {world_name}."

    sentences = [
        sentence_start + "[MASK]" + sentence_end,
        sentence_start + "[MASK] of" + sentence_end,
        sentence_start + "[MASK] in" + sentence_end,
        sentence_start + "[MASK] to" + sentence_end,
    ]

    results = []
    scores = np.zeros((len(sentences),))
    for i, sentence in enumerate(sentences):
        result = unmasker(sentence, top_k=1)[0]
        results.append(result)
        scores[i] = result["score"]

    sentence = results[scores.argmax()]["sequence"]

    return sentence


def human_name(block_id: str) -> str:
    """
    Convert 'minecraft:oak_planks' -> 'oak planks'
    """
    return block_id.replace("minecraft:", "").replace("_", " ")

def get_neighbor_calculation_sentence_positions(
    block_name: str,
    world_name: str,
    is_world_plural: bool,
    neighbors: List[Dict],
    out_of_bounds_token: str = "__OUT_OF_BOUNDS__"
) -> str:
    """
    Build a sentence like:
    "The center is cobblestone. It is surrounded by 3 dirt blocks,
    5 grass blocks and 1 air block in this village."
    """

    # The center always refers to one already-cleaned block name.
    center_name = block_name
    first_sentence = f"The center block is {center_name}."

    # Count valid neighbors by their human-readable block name.
    counts = Counter()
    for nb in neighbors:
        nb_block = nb["block_name"]
        if nb_block == out_of_bounds_token or nb_block == "none":
            continue

        nb_clean = human_name(nb_block)
        counts[nb_clean] += 1

    # Fall back to world membership when no valid neighbors are available.
    if not counts:
        if is_world_plural:
            return first_sentence + f" It is part of these {world_name}."
        else:
            return first_sentence + f" It is part of this {world_name}."

    # Sort by count and name to keep generated descriptions deterministic.
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    phrases = []
    for nb_name, cnt in items:
        block_word = "block" if cnt == 1 else "blocks"
        phrases.append(f"{cnt} {nb_name} {block_word}")

    # Join phrases using natural English list punctuation.
    if len(phrases) == 1:
        neighbors_str = phrases[0]
    elif len(phrases) == 2:
        neighbors_str = f"{phrases[0]} and {phrases[1]}"
    else:
        neighbors_str = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    if is_world_plural:
        world_phrase = f"these {world_name}"
    else:
        world_phrase = f"this {world_name}"

    second_sentence = f" It is surrounded by {neighbors_str} in {world_phrase}."

    return first_sentence + second_sentence


def group_from_label(pos_label: str) -> str:
    """Group a relative position into ``above``, ``below``, or ``side``."""
    if pos_label.startswith("up"):
        return "above"
    if pos_label.startswith("down"):
        return "below"
    return "side"


def accumulate_neighbor_stats(level: torch.Tensor,
                              coords,
                              token_list,
                              neighbor_info):
    """Count neighboring block types by direction for each center block.

    ``level`` has shape ``(1, C, Y, Z, X)`` and ``coords`` follows the
    ``((y0, y1), (z0, z1), (x0, x1))`` convention used by ``Config.coords``.
    The result is indexed as ``stats[center_block][direction][neighbor]``.
    """
    (y0, y1), (z0, z1), (x0, x1) = coords

    stats = defaultdict(lambda: defaultdict(Counter))

    # level: (1, C, Y, Z, X)
    _, C, Y, Z, X = level.shape

    for (j, k, l), neigh_list in neighbor_info.items():
        iy = j - y0
        iz = k - z0
        ix = l - x0

        if not (0 <= iy < Y and 0 <= iz < Z and 0 <= ix < X):
            continue

        # Recover the center token from the one-hot channel dimension.
        center_idx = level[0, :, iy, iz, ix].argmax().item()
        center_block = token_list[center_idx]

        for nb in neigh_list:
            nb_block = nb["block_name"]
            if nb_block == "__OUT_OF_BOUNDS__":
                continue

            pos_label = nb["pos_label"]
            group = group_from_label(pos_label)  # 'above' / 'below' / 'side'
            stats[center_block][group][nb_block] += 1

    return stats

def clean_block_name(name: str) -> str:
    return name.replace("minecraft:", "").replace("_", " ")

def restore_block_name(name: str) -> str:
    return "minecraft:" + name.replace(" ", "_")

def build_context_sentence_for_block(block_name: str,
                                     world_name: str,
                                     stats_for_block: dict) -> str:
    """Describe a block using its three directional neighbor counters."""
    center_human = clean_block_name(block_name)

    sentence = f"The center block is {center_human}."

    if not stats_for_block:
        return sentence + f" It is part of this {world_name}."

    # Combine direction-specific counts before selecting common neighbors.
    total_counter = Counter()
    for group_counter in stats_for_block.values():
        total_counter.update(group_counter)

    filtered = []
    for nb_block, cnt in total_counter.most_common():
        if nb_block is None:
            continue
        nb_clean = clean_block_name(nb_block)
        if nb_clean == "air":
            continue
        filtered.append((nb_clean, cnt))

    if not filtered:
        return sentence + f" It is part of this {world_name}."

    top_neighbors = [name for name, _ in filtered[:3]]

    # Format at most three neighbors as a natural-language list.
    if len(top_neighbors) == 1:
        neighbors_str = top_neighbors[0]
    elif len(top_neighbors) == 2:
        neighbors_str = f"{top_neighbors[0]} and {top_neighbors[1]}"
    else:
        neighbors_str = ", ".join(top_neighbors[:-1]) + f", and {top_neighbors[-1]}"

    sentence += f" It is surrounded by {neighbors_str} in this {world_name}."
    return sentence
