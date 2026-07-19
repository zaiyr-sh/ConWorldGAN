import os
import pprint
from timeit import default_timer
from typing import List

import inflect
import numpy as np
import torch
import transformers
from torch import Tensor
from tqdm import tqdm
from sklearn.decomposition import PCA

from config import Config, parse_args
from constants import (
    BERT_MODEL_NAME,
    REGION_NAMES,
    SUB_COORDS,
    WORLD_LABELS_PLURAL,
)
from minecraft.block_grammar import (
    accumulate_neighbor_stats,
    build_context_sentence_for_block,
    clean_block_name,
    get_neighbor_calculation_sentence_positions,
    get_sentence,
)
from minecraft.level_utils import read_map
from utils import save_pkl


# ---------- Utilities ----------

def get_bert_objects(model_name: str, opt: Config):
    model = transformers.BertModel.from_pretrained(model_name).to(opt.device)
    model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    unmasker = transformers.pipeline("fill-mask", model=model_name)
    return model, tokenizer, unmasker

def compress_dim(opt: Config, natural_tokens: Tensor) -> Tensor:
    tokens_cpu = natural_tokens.detach().cpu().numpy()

    if opt.repr_dim is None or opt.repr_dim <= 0:
        pca_full = PCA()
        pca_full.fit(tokens_cpu)
        explained = pca_full.explained_variance_ratio_
        cum_explained = np.cumsum(explained)

        threshold = 0.99
        n_components = np.searchsorted(cum_explained, threshold) + 1
        opt.repr_dim = n_components
        print(f"[PCA] auto n_components={n_components} for {threshold * 100:.1f}% variance")
    else:
        n_components = opt.repr_dim

    pca = PCA(n_components=n_components)
    embedding_cpu = pca.fit_transform(tokens_cpu)  # shape: (n_tokens, repr_dim)

    embedding = torch.tensor(embedding_cpu, dtype=torch.float32).to(opt.device)

    print(
        f"[PCA] Fitting PCA for {natural_tokens.shape[0]} tokens, input dim={natural_tokens.shape[1]}, output dim={opt.repr_dim}")

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"Cum var: {cum_var}")

    return embedding

# ---------- Main flow ----------

if __name__ == '__main__':
    batch_size = 128
    times = {'tokenize': 0, 'to_gpu': 0, 'forward': 0, 'other': 0}

    inflect = inflect.engine()
    # This script creates the representation files, so they cannot be loaded yet.
    opt = parse_args(load_representations=False)
    opt.repr_type = None
    model, tokenizer, unmasker = get_bert_objects(BERT_MODEL_NAME, opt)
    level = read_map(opt)

    if opt.neighbors_type == "global":
        stats = accumulate_neighbor_stats(
            level=level,
            coords=opt.coords,
            token_list=opt.token_list,
            neighbor_info=opt.neighbor_info,
        )

    if opt.device.type == "cuda":
        # enables TF32 on Ampere+ and better matmul perf
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    for idx, region_name in enumerate(REGION_NAMES):
        prepath = os.path.join(opt.representation_dir, region_name)
        os.makedirs(prepath, exist_ok=True)
        print(f"[region] {region_name}")
        opt.input_area_name = region_name
        opt.sub_coords = SUB_COORDS[region_name]

        print(opt)
        token_list = opt.token_list
        token_names: List[str] = []
        clean_names: List[str] = []

        if opt.neighbors_type == "global":
            for block_name in token_list:
                print(block_name)
                world_name = WORLD_LABELS_PLURAL[idx][0]
                if clean_block_name(block_name) == "air":
                    sent = "The center block is air."
                else:
                    sent = build_context_sentence_for_block(
                        block_name=block_name,
                        world_name=world_name,
                        stats_for_block=stats.get(block_name, {}),
                    )
                token_names.append(sent)
                clean_names.append(block_name)
                print(f"Sentence: '{token_names[-1]}'")
        elif opt.neighbors_type == "local":
            for value in tqdm(opt.neighbor_info.values()):
                clean_token = value[0]["block_name"].replace("minecraft:", "").replace("_", " ")

                if clean_token == "air":
                    if any(s.startswith("air") for s in clean_names):
                        continue
                    token_names.append("This air is part of this village.")
                    clean_names.append("air")
                    continue

                token_names.append(
                    get_neighbor_calculation_sentence_positions(
                        clean_token,
                        WORLD_LABELS_PLURAL[idx][0],
                        WORLD_LABELS_PLURAL[idx][1],
                        value,
                    )
                )
                j = value[0]["y"]
                k = value[0]["z"]
                l = value[0]["x"]
                clean_token = clean_token.replace(" ", "_")
                clean_names.append(f"{clean_token}_{(j, k, l)}")
        else:
            for token in token_list:
                clean_token = token.replace("minecraft:", "").replace("_", " ")
                if isinstance(inflect.singular_noun(clean_token), bool) or (clean_token.find("grass") >= 0):
                    is_plural = False
                else:
                    is_plural = True
                token_names.append(get_sentence(clean_token, is_plural, WORLD_LABELS_PLURAL[idx][0], WORLD_LABELS_PLURAL[idx][1], unmasker))
                clean_names.append(token)
                print(f"Sentence: '{token_names[-1]}'")

        token_list = clean_names
        natural_token_dict = {}

        with torch.no_grad():
            for i in tqdm(range(0, len(token_names), batch_size)):
                batched_token_name = token_names[i:i + batch_size]
                batched_token = token_list[i:i + batch_size]

                start = default_timer()
                ids = tokenizer(batched_token_name, padding=True, truncation=True, return_tensors="pt")
                end = default_timer()
                times['tokenize'] += end - start

                start = default_timer()
                ids = {k: v.to(opt.device) for k, v in ids.items()}
                if opt.device != torch.device('cpu'):
                    torch.cuda.synchronize()
                end = default_timer()
                times['to_gpu'] += end - start

                start = default_timer()
                bert_output = model.forward(**ids, output_hidden_states=True)
                if opt.device != torch.device('cpu'):
                    torch.cuda.synchronize()
                end = default_timer()
                times['forward'] += end - start

                start = default_timer()
                final_layer_embeddings = bert_output.last_hidden_state[:, 0]
                end = default_timer()
                times['other'] += end - start

                start = default_timer()
                for token, embedding in zip(batched_token, final_layer_embeddings.to('cpu')):
                    natural_token_dict[token] = torch.tensor(embedding)
                end = default_timer()
                times['other'] += end - start

        print(bert_output.last_hidden_state[:, 0].shape)
        print('batched')
        pprint.pprint(times)
        print(len(natural_token_dict.keys()))

        if opt.neighbors_type is None:
            save_pkl(natural_token_dict, f"natural_representations", prepath)
        else:
            dim = f"_{opt.repr_dim}" if opt.repr_dim else ""
            save_pkl(natural_token_dict,f"natural_representations_neighbors{dim}", prepath)


        natural_tokens = torch.stack(list(natural_token_dict.values()))
        natural_tokens = natural_tokens.to(device=opt.device, non_blocking=True)

        embedding = compress_dim(opt, natural_tokens)

        natural_token_dict_small_no_norm = {}
        natural_token_dict_small = {}

        for token_name, e in tqdm(zip(token_list, embedding)):
            natural_token_dict_small_no_norm[token_name] = e
            natural_token_dict_small[token_name] = e / torch.norm(e, p=2)

        if opt.neighbors_type is None:
            save_pkl(natural_token_dict_small_no_norm,
                     f"natural_representations_small_no_norm_{opt.repr_dim}", prepath)
            save_pkl(natural_token_dict_small,
                     f"natural_representations_small_{opt.repr_dim}", prepath)
        else:
            save_pkl(natural_token_dict_small_no_norm,
                     f"natural_representations_small_neighbors_no_norm_{opt.repr_dim}", prepath)
            save_pkl(natural_token_dict_small,
                     f"natural_representations_small_neighbors_{opt.repr_dim}", prepath)
