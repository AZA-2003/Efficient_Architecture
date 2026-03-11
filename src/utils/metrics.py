import transformers
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

'''

'''
def calculate_memory_footprint(model:transformers.models,):
  total_params = sum(p.numel() for p in model.parameters())
  return round((total_params*4)/(1024*1024*1024),3)

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
                          do_sample=False,)
  time_elapsed = time.time() - start
  new_token_length = outputs.shape[1] - initial_length
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
  seq_len = example.input_ids.size(1)

  nll_sum = 0.0
  n_tokens = 0
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride),leave=False):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = example.input_ids[:, begin_loc:end_loc].to("cuda")
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)

          # loss is calculated using CrossEntropyLoss which averages over valid labels
          # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
          # to the left by 1.
          neg_log_likelihood = outputs.loss

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


def get_metrics(model: transformers.models,
                  dataloader: DataLoader,
                  read_length: int,
                  gen_length: int):
  # mem = calculate_memory_footprint(model)
  ttft = []
  tps = []
  ppl = []
  model = model.to("cuda")
  for batch in dataloader:
    batch = batch.to("cuda")
    ttft.append(time_to_first_token(model,batch))
    tps.append(tokens_per_second(model,batch,max_length = read_length+gen_length, stride=read_length))
    ppl.append(calculate_perplexity(model,batch))
  return sum(ttft)/len(ttft), sum(tps)/len(tps), sum(ppl)/len(ppl)


    

  

  

