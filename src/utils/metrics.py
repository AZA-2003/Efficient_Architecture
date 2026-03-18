import transformers
import torch
import gc
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time
from tqdm import tqdm

'''

'''
def calculate_memory_footprint(model:transformers.models,):
  #total_params = sum(p.numel() for p in model.parameters())
  total_param_bytes = model.get_memory_footprint()
  return round((total_param_bytes)/(1024*1024*1024),3)

'''

'''
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

'''

'''
def time_to_first_token(model:transformers.models,
                        example,):
  start = time.time()
  model(**example)
  time_elapsed = time.time() - start
  return time_elapsed

'''

'''
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

def model_profiler(model: transformers.models,
                  example):
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model = model.to("cuda")
    example = example.to("cuda")
    model(**example)
  
  prof.export_chrome_trace("trace.json")
  return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
