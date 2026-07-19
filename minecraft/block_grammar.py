from typing import List, Dict

import numpy as np
import inflect
import torch

from constants import ORDER, SHORT, HOUSE_BLOCKS

N_SENTENCES = 6  # len(sentence_mid)


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

    # sentence_mid = [
    #     "part of",
    #     "hidden in",
    #     "on top of",
    #     "below",
    #     "next to"
    # ]

    # sentence = sentence_start + sentence_mid[n_sentence] + sentence_end
    return sentence


def get_neighbor_sentence(block_name, is_block_plural, world_name, is_world_plural, neighbors, n_plurals, unmasker):
    if is_block_plural:
        sentence_start = f"These {block_name} are "
    else:
        sentence_start = f"This {block_name} is "

    sentence_mid = "surrounded by"
    if len(neighbors) > 1:
        for i, n in enumerate(neighbors[:-1]):
            if n_plurals[i]:
                sentence_mid = sentence_mid + f" {n},"
            else:
                sentence_mid = sentence_mid + f" a {n},"
        if n_plurals[-1]:
            sentence_mid = sentence_mid + f" and {neighbors[-1]}. It is "
        else:
            sentence_mid = sentence_mid + f" and a {neighbors[-1]}. It is "
    else:
        if neighbors[0] == block_name:
            # surrounded by only itself
            sentence_mid = ""
        else:
            # only one other block in neighborhood
            if n_plurals[0]:
                sentence_mid = f"next to {neighbors[0]}. It is "
            else:
                sentence_mid = f"next to a {neighbors[0]}. It is "

    if is_world_plural:
        sentence_end = f" these {world_name}."
    else:
        sentence_end = f" this {world_name}."

    sentences = [
        sentence_start + sentence_mid + "[MASK]" + sentence_end,
        sentence_start + sentence_mid + "[MASK] of" + sentence_end,
        sentence_start + sentence_mid + "[MASK] in" + sentence_end,
        sentence_start + sentence_mid + "[MASK] to" + sentence_end,
    ]

    results = []
    scores = np.zeros((len(sentences),))
    for i, sentence in enumerate(sentences):
        result = unmasker(sentence, top_k=1)[0]
        results.append(result)
        scores[i] = result["score"]

    sentence = results[scores.argmax()]["sequence"]
    return sentence

inflect_engine = inflect.engine()

def human_name(block_id: str) -> str:
    """
    Convert 'minecraft:oak_planks' -> 'oak planks'
    """
    return block_id.replace("minecraft:", "").replace("_", " ")

def get_neighbor_dictionary_positions(
    block_name: str,
    is_block_plural: bool,
    world_name: str,
    is_world_plural: bool,
    neighbors: List[Dict],
    unmasker,
    out_of_bounds_token="__OUT_OF_BOUNDS__"
):
    return schema_string(block_name, world_name, neighbors, oob=out_of_bounds_token)

def get_neighbor_calculation_sentence_positions(
    block_name: str,
    is_block_plural: bool,
    world_name: str,
    is_world_plural: bool,
    neighbors: List[Dict],
    unmasker,  # not used anymore, kept for compat
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
        if cnt == 1:
            # "1 dirt block"
            block_word = "block"
            block_word_pl = block_word  # "1 block"
        else:
            block_word = "block"
            block_word_pl = inflect_engine.plural(block_word)  # "blocks"

        phrases.append(f"{cnt} {nb_name} {block_word_pl}")

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

def get_neighbor_sentence_positions(
    block_name: str,
    is_block_plural: bool,
    world_name: str,
    is_world_plural: bool,
    neighbors: List[Dict],
    unmasker,
    out_of_bounds_token="__OUT_OF_BOUNDS__"
):

    if is_block_plural:
        sentence_start = f"These {block_name} are "
    else:
        sentence_start = f"This {block_name} is "

    # Build middle phrase describing neighbors
    sentence_mid = "surrounded by "

    described = []
    for nb in neighbors:
        nb_block = nb["block_name"]
        nb_label = nb["pos_label"]  # "up", "front-left", etc.

        if nb_block == out_of_bounds_token or nb_label == "center":
            continue  # skip outside area

        nb_name = human_name(nb_block)

        # Plurality check using inflect
        plural = inflect_engine.singular_noun(nb_name)
        is_plural = not (plural is False)

        if is_plural:
            described.append(f"{nb_name} on the {nb_label} side")
        else:
            described.append(f"a {nb_name} on the {nb_label} side")

    if len(described) > 1:
        sentence_mid += ", ".join(described[:-1])
        sentence_mid += f", and {described[-1]} "
    elif len(described) == 1:
        sentence_mid = f"next to {described[0]} "
    else:
        sentence_mid = ""  # No useful neighbors

    # World name
    if is_world_plural:
        sentence_end = f" these {world_name}."
    else:
        sentence_end = f" this {world_name}."

    return sentence_start + sentence_mid + "in" + sentence_end

def schema_string(center_block, world_name, neighbors, oob="__OUT_OF_BOUNDS__"):
    parts = [f"CENTER={center_block}", f"WORLD={world_name}"]
    by_label = {d["pos_label"]: d["block_name"] for d in neighbors if d["block_name"] != oob}
    for label in ORDER:
        block_name = by_label.get(label)
        if block_name and block_name != "none":
            parts.append(f"{SHORT[label]}={block_name}")
    return "; ".join(parts) + "."

def schema_string_with_3_neighbors(center_block: str, world_name: str, neighbors: List[Dict], oob: str = "__OUT_OF_BOUNDS__") -> str:
    """
    Build a simple natural sentence like:
    "The CENTER block is oak planks. Around it you often see water, grass, and air."
    """

    def clean(name: str) -> str:
        # remove minecraft: prefix and turn underscores into spaces
        return name.replace("minecraft:", "").replace("_", " ")

    center_clean = clean(center_block)

    # collect neighbor block names (ignore out-of-bounds and "none")
    raw_neighbor_names = []
    for d in neighbors:
        bname = d["block_name"]
        if bname == oob or bname == "none":
            continue
        raw_neighbor_names.append(clean(bname))

    # deduplicate while preserving order
    seen = set()
    neighbor_names = []
    for n in raw_neighbor_names:
        if n not in seen:
            seen.add(n)
            neighbor_names.append(n)

    # first sentence: describe center
    sentence_start = f"The CENTER block is {center_clean}."

    # if we have no valid neighbors, just return the first sentence
    if not neighbor_names:
        return sentence_start

    # build "water, grass, and air" / "water and grass" part
    if len(neighbor_names) == 1:
        neighbors_str = neighbor_names[0]
    elif len(neighbor_names) == 2:
        neighbors_str = f"{neighbor_names[0]} and {neighbor_names[1]}"
    else:
        neighbors_str = ", ".join(neighbor_names[:-1]) + f", and {neighbor_names[-1]}"

    sentence_mid = f" Around it you often see {neighbors_str}."

    return sentence_start + sentence_mid


from collections import defaultdict, Counter

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

def _most_common_non_air(counter):
    """Return the most common non-air block, or ``None`` when unavailable."""
    if counter is None:
        return None

    for block_name, _ in counter.most_common():
        # raw: "minecraft:air" -> clean: "air"
        if clean_block_name(block_name) != "air":
            return block_name
    return None

def is_house_block(clean_name: str) -> bool:
    return clean_name in HOUSE_BLOCKS

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

def build_context_sentence_for_block_with_house(block_name: str,
                                     stats_for_block: dict) -> str:
    """Describe a block's neighbors and whether it belongs to a house."""
    center_human = clean_block_name(block_name)

    sentence = f"The center block is {center_human}."

    # Without neighbor statistics, return only the house classification.
    if not stats_for_block:
        house_part = (
            " This block is part of a house."
            if is_house_block(center_human)
            else " This block is not part of a house."
        )
        return sentence + house_part

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
        house_part = (
            " This block is part of a house."
            if is_house_block(center_human)
            else " This block is not part of a house."
        )
        return sentence + house_part

    top_neighbors = [name for name, _ in filtered[:3]]

    if len(top_neighbors) == 1:
        neighbors_str = top_neighbors[0]
    elif len(top_neighbors) == 2:
        neighbors_str = f"{top_neighbors[0]} and {top_neighbors[1]}"
    else:
        neighbors_str = ", ".join(top_neighbors[:-1]) + f", and {top_neighbors[-1]}"

    sentence += f" It is surrounded by {neighbors_str}."

    house_part = (
        " This block is part of a house."
        if is_house_block(center_human)
        else " This block is not part of a house."
    )
    sentence += house_part

    return sentence
