import transformers
import torch
from torch.utils.data import DataLoader
import sys
import json
from pathlib import Path
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
               data,
               json_path: str | Path | None = None):
    """
    Run the benchmark suite.

    If `json_path` is provided, results are incrementally written after each successful
    (read_len, gen_len) point so that if an OOM happens mid-run, progress is preserved.

    Notes:
    - On CUDA OOM, this function stops further points for the current model
      (keeps already computed points) and returns partial results.
    - Failed points are omitted from the returned dict so plotting helpers can skip them
      without requiring all (read_len, gen_len) keys to exist.
    """
    metrics = {}
    print(calculate_memory_footprint(model))

    out_path = Path(json_path) if json_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize output early so incremental progress has a file to write to.
        with open(out_path, "w") as f:
            json.dump({}, f)

    max_ctx = _get_model_max_context(model)
    if max_ctx is not None:
        print(f"Max context supported by model: {max_ctx}")

    def _dump_metrics_partial():
        if out_path is None:
            return
        serializable = {}
        for (r, g), vals in metrics.items():
            serializable[f"({r},{g})"] = vals
        with open(out_path, "w") as f:
            json.dump(serializable, f)

    for read_len, gen_len in test_suite:
        if max_ctx is not None and read_len > max_ctx:
            print(f"{read_len}/{gen_len} skipped (read_len > max context {max_ctx})")
            continue

        dataloader = data.process(20, read_len, 1,)
        try:
            pm, ttft, tps, ppl = get_metrics(model, dataloader, read_len, gen_len)
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM at {read_len}/{gen_len}; returning partial results.")
            _dump_metrics_partial()
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                print(f"RuntimeError OOM at {read_len}/{gen_len}; returning partial results.")
                _dump_metrics_partial()
                break
            raise

        metrics[(read_len, gen_len)] = {
            "Peak Mem.": pm,
            "TTFT": ttft,
            "TPS": tps,
            "PPL": ppl,
        }

        _dump_metrics_partial()

        print(f"{read_len}/{gen_len}")
        print(f"Peak Memory: {pm}")
        print(f"Time to First Token: {ttft}")
        print(f"Tokens per second: {tps}")
        print(f"Perplexity: {ppl}")

    return metrics