import transformers
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
from Efficient_Architecture.src.utils.metrics import *

from test_vectors import benchmark_test_suite

# Backwards-compatible name used throughout notebooks / code.
test_suite = benchmark_test_suite

def _get_model_max_context(model) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return None

def test_model(model: transformers.models,
               data):
    metrics = {}
    print(calculate_memory_footprint(model))
    max_ctx = _get_model_max_context(model)
    if max_ctx is not None:
        print(f"Max context supported by model: {max_ctx}")
    for read_len,gen_len in test_suite:
        metrics[(read_len,gen_len)] = {}
        if max_ctx is not None and read_len > max_ctx:
            print(f"{read_len}/{gen_len} skipped (read_len > max context {max_ctx})")
            continue
        dataloader = data.process(20,read_len,1,)
        pm,ttft,tps,ppl = get_metrics(model, dataloader, read_len, gen_len)
        
        metrics[(read_len,gen_len)]["Peak Mem."] = pm
        metrics[(read_len,gen_len)]["TTFT"] = ttft
        metrics[(read_len,gen_len)]["TPS"] = tps
        metrics[(read_len,gen_len)]["PPL"] = ppl
        
        print(f"{read_len}/{gen_len}")
        print(f"Peak Memory: {pm}")
        print(f"Time to First Token: {ttft}")
        print(f"Tokens per second: {tps}")
        print(f"Perplexity: {ppl}")
    return metrics