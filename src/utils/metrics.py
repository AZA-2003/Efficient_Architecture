import gc
import time
from pathlib import Path

import torch
import transformers
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# Utilities
# =============================================================================

def count_params(module: torch.nn.Module) -> int:
  return sum(p.numel() for p in module.parameters())


def non_embedding_params(model: torch.nn.Module) -> int:
  total = count_params(model)

  inp = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
  inp_n = count_params(inp) if inp is not None else 0

  out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
  out_n = 0 if (out is None or out is inp) else count_params(out)

  return total - inp_n - out_n


def _infer_model_device(model: torch.nn.Module) -> torch.device:
  """Infer device from the first model parameter (handles common placement cases)."""
  return next(model.parameters()).device


def safe_move_batch_to_device(batch, device: torch.device):
  """
  Move only tensor values to the target device.
  Avoids calling BatchEncoding.to(dtype=...) which can raise errors.
  """
  if isinstance(batch, dict):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

  if hasattr(batch, "items"):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

  raise TypeError(f"Unsupported batch type for device move: {type(batch)}")


# =============================================================================
# Original metrics (benchmarking / evaluation)
# =============================================================================

"""

"""
def calculate_memory_footprint(model:transformers.models,):
  #total_params = sum(p.numel() for p in model.parameters())
  total_param_bytes = model.get_memory_footprint()
  return round((total_param_bytes)/(1024*1024*1024),3)

"""

"""
def tokens_per_second(model:transformers.models,
                      example,
                      num_tokens: int):
  start = time.time()
  initial_length = example["input_ids"].shape[1]
  outputs = model.generate(**example,
                          max_new_tokens=num_tokens,
                          num_beams=4,
                          do_sample=False,).cpu()
  time_elapsed = time.time() - start
  new_token_length = outputs.shape[1]
  del outputs
  return new_token_length/time_elapsed

"""

"""
def time_to_first_token(model:transformers.models,
                        example,):
  start = time.time()
  model(**example)
  time_elapsed = time.time() - start
  return time_elapsed

"""

"""
def calculate_perplexity(model:transformers.models,
                        example,
                        max_length = 1024,
                        stride = 16):
  #max_length = model.config.n_positions
  max_length = max_length
  stride = stride
  device = next(model.parameters()).device
  input_ids_full = example["input_ids"] if isinstance(example, dict) else example.input_ids
  seq_len = input_ids_full.size(1)

  nll_sum = 0.0
  n_tokens = 0
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride),leave=False):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = input_ids_full[:, begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)

          # loss is calculated using CrossEntropyLoss which averages over valid labels
          # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
          # to the left by 1.
          neg_log_likelihood = outputs.loss
          del outputs

      # Accumulate the total negative log-likelihood and the total number of tokens
      num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
      batch_size = target_ids.size(0)
      num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
      nll_sum += neg_log_likelihood * num_loss_tokens
      n_tokens += num_loss_tokens

      prev_end_loc = end_loc
      if end_loc == seq_len:
          break

  avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
  ppl = torch.exp(avg_nll)
  return ppl.item()

def peak_memory(model:transformers.models,
                example,):
  if torch.cuda.is_available():
    torch.cuda.memory.reset_peak_memory_stats()
  model(**example)
  if torch.cuda.is_available():
    peak_mem = torch.cuda.memory.max_memory_allocated()
    torch.cuda.memory.reset_peak_memory_stats()
    return round(peak_mem/(1024**3),3)
  return 0.0

def get_metrics(model: transformers.models,
                  dataloader: DataLoader,
                  read_length: int,
                  gen_length: int):
  # mem = calculate_memory_footprint(model)
  ttft = []
  tps = []
  ppl = []
  pm = []
  if torch.cuda.is_available():
    model = model.to("cuda")
  device = next(model.parameters()).device
  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      # Prefer HF BatchEncoding.to(device) when available; it handles nested structures safely.
      if hasattr(batch, "to"):
        batch = batch.to(device)
      else:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
      #pm.append(peak_memory(model,batch))
      ttft.append(time_to_first_token(model,batch))
      if torch.cuda.is_available():
        torch.cuda.memory.reset_peak_memory_stats()
      tps.append(tokens_per_second(model,batch,gen_length))
      if torch.cuda.is_available():
        pm.append(round(torch.cuda.memory.max_memory_allocated()/(1024**3),3))
        torch.cuda.memory.reset_peak_memory_stats()
      else:
        pm.append(0.0)
      ppl.append(calculate_perplexity(model,batch,max_length = read_length+gen_length, stride=read_length))
      del batch
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      gc.collect()
  return sum(pm)/len(pm), sum(ttft)/len(ttft), sum(tps)/len(tps), sum(ppl)/len(ppl)


# =============================================================================
# Profiling (torch.profiler, prefill/decode traces, chrome trace export)
# =============================================================================

# Centralized vector definitions
from test_vectors import profiling_test_suite


def profile_prefill(
  model: transformers.models,
  batch,
  use_profiler: bool = True,
  trace_path: str | None = None,
):
  """
  Prefill = one forward pass over the prompt.
  Returns (outputs, timing_s, cuda_peak_gb, cpu_peak_mb, profiler_tables)
  """
  model.eval()
  device = _infer_model_device(model)
  batch = safe_move_batch_to_device(batch, device)

  activities = (
    [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    if (use_profiler and torch.cuda.is_available())
    else ([ProfilerActivity.CPU] if use_profiler else [])
  )

  import tracemalloc
  tracemalloc.start()
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

  profiler_tables = {}

  with torch.no_grad():
    if use_profiler:
      with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
      ) as prof:
        t0 = time.time()
        outputs = model(**batch, use_cache=True)
        t1 = time.time()

      if trace_path is not None:
        prof.export_chrome_trace(trace_path)

      # Capture key_averages tables as strings (so callers can write them to logs/files).
      # Keep it conservative in size to avoid giant logs.
      try:
        time_sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        profiler_tables["prefill_key_averages_time_table"] = prof.key_averages().table(
          sort_by=time_sort,
          row_limit=50,
        )
      except Exception as e:
        profiler_tables["prefill_key_averages_time_table_error"] = str(e)

      try:
        profiler_tables["prefill_key_averages_flops_table"] = prof.key_averages().table(sort_by="flops", row_limit=20)
      except Exception as e:
        profiler_tables["prefill_key_averages_flops_table_error"] = str(e)

      try:
        profiler_tables["prefill_key_averages_cuda_mem_table"] = prof.key_averages().table(
          sort_by="self_cuda_memory_usage",
          row_limit=20,
        )
      except Exception as e:
        profiler_tables["prefill_key_averages_cuda_mem_table_error"] = str(e)

    else:
      t0 = time.time()
      outputs = model(**batch, use_cache=True)
      t1 = time.time()

  _, peak_cpu_bytes = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  peak_gpu_gb = 0.0
  if torch.cuda.is_available():
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gpu_gb = round(peak_bytes / (1024 ** 3), 3)
    torch.cuda.reset_peak_memory_stats()

  return outputs, (t1 - t0), peak_gpu_gb, round(peak_cpu_bytes / (1024 ** 2), 3), profiler_tables


def profile_decode_with_past(
  model: transformers.models,
  batch,
  past_key_values,
  decode_steps: int,
  use_profiler: bool = True,
  trace_path: str | None = None,
):
  """
  Decode = iterative steps using past_key_values.
  Returns (generated_tokens, timing_s, cuda_peak_gb, cpu_peak_mb, profiler_tables)
  """
  model.eval()
  device = _infer_model_device(model)
  batch = safe_move_batch_to_device(batch, device)

  input_ids = batch["input_ids"]
  next_token = input_ids[:, -1:].contiguous()

  pkv = past_key_values
  generated = []

  activities = (
    [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    if (use_profiler and torch.cuda.is_available())
    else ([ProfilerActivity.CPU] if use_profiler else [])
  )

  import tracemalloc
  tracemalloc.start()
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

  profiler_tables = {}

  with torch.no_grad():
    if use_profiler:
      with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
      ) as prof:
        t0 = time.time()
        for _ in range(decode_steps):
          outputs = model(
            input_ids=next_token,
            past_key_values=pkv,
            use_cache=True,
          )
          logits = outputs.logits
          pkv = outputs.past_key_values
          next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
          generated.append(next_token)
        t1 = time.time()

      if trace_path is not None:
        prof.export_chrome_trace(trace_path)

      try:
        time_sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        profiler_tables["decode_key_averages_time_table"] = prof.key_averages().table(
          sort_by=time_sort,
          row_limit=50,
        )
      except Exception as e:
        profiler_tables["decode_key_averages_time_table_error"] = str(e)

      try:
        profiler_tables["decode_key_averages_flops_table"] = prof.key_averages().table(sort_by="flops", row_limit=20)
      except Exception as e:
        profiler_tables["decode_key_averages_flops_table_error"] = str(e)

      try:
        profiler_tables["decode_key_averages_cuda_mem_table"] = prof.key_averages().table(
          sort_by="self_cuda_memory_usage",
          row_limit=20,
        )
      except Exception as e:
        profiler_tables["decode_key_averages_cuda_mem_table_error"] = str(e)

    else:
      t0 = time.time()
      for _ in range(decode_steps):
        outputs = model(
          input_ids=next_token,
          past_key_values=pkv,
          use_cache=True,
        )
        logits = outputs.logits
        pkv = outputs.past_key_values
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
      t1 = time.time()

  _, peak_cpu_bytes = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  peak_gpu_gb = 0.0
  if torch.cuda.is_available():
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gpu_gb = round(peak_bytes / (1024 ** 3), 3)
    torch.cuda.reset_peak_memory_stats()

  return generated, (t1 - t0), peak_gpu_gb, round(peak_cpu_bytes / (1024 ** 2), 3), profiler_tables


def profile_prefill_decode(
  model: transformers.models,
  batch,
  decode_steps: int,
  traces_dir: str | None = None,
  trace_prefix: str | None = None,
):
  """
  Profile prefill and decode phases and return a flat dict of results.
  """
  traces_dir = traces_dir or "."
  Path(traces_dir).mkdir(parents=True, exist_ok=True)
  trace_prefix = trace_prefix or "trace"

  prefill_trace = str(Path(traces_dir) / f"{trace_prefix}_prefill.json")
  decode_trace = str(Path(traces_dir) / f"{trace_prefix}_decode.json")

  prefill_out, prefill_s, prefill_peak_gb, prefill_cpu_peak_mb, prefill_tables = profile_prefill(
    model,
    batch,
    use_profiler=True,
    trace_path=prefill_trace,
  )
  past = prefill_out.past_key_values

  _, decode_s, decode_peak_gb, decode_cpu_peak_mb, decode_tables = profile_decode_with_past(
    model,
    batch,
    past_key_values=past,
    decode_steps=decode_steps,
    use_profiler=True,
    trace_path=decode_trace,
  )

  return {
    "prefill_s": prefill_s,
    "decode_s": decode_s,
    "prefill_peak_gpu_gb": prefill_peak_gb,
    "decode_peak_gpu_gb": decode_peak_gb,
    "prefill_cpu_peak_mb": prefill_cpu_peak_mb,
    "decode_cpu_peak_mb": decode_cpu_peak_mb,
    **prefill_tables,
    **decode_tables,
  }


def model_profiler(model: transformers.models,
                  example,
                  trace_path: str = "trace.json"):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = model.to(device)
  example = safe_move_batch_to_device(example, device)

  activities = [ProfilerActivity.CPU]
  if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

  with profile(activities=activities) as prof:
    with torch.no_grad():
      model(**example)

  prof.export_chrome_trace(trace_path)
  return prof.key_averages().table(sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total", row_limit=-1)
