import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import datetime
import json
import random
from tqdm import tqdm
from typing import Union

import numpy as np

import torch
torch.set_float32_matmul_precision('high')
torch.utils.backcompat.broadcast_warning.enabled=True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast # type: ignore #

from model.model import MultiGeometryGPT
from config.train_config import TrainConfig
from config.model_config import MultiGeomGPTConfig
from utils.loader import DistributedDataLoader
from utils.muon import Muon
from prodigyopt import Prodigy


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def create_run_id(train_config: TrainConfig, model_config: MultiGeomGPTConfig, dataset_name: str, timestamp: datetime.datetime):
    """Create a run identifier."""
    # Aliases for common configurations
    dataset_aliases = {
        'TinyStories': 'ts',
        'TinyStoriesChar': 'tsc',
        'FineWeb': 'fw',
        'Wikitext2': 'wt2',
        'Wikitext103': 'wt103',
    }
    mode_aliases = {
        'euc': 'e',
        'hyp': 'h',
        'sph': 's',
    }
    
    # Get date and time components
    date = timestamp.strftime('%m.%d') 
    seconds_since_midnight = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).seconds
    
    # Get architecture configuration
    head, n1, n2, n3 = mode_aliases[model_config.lm_head_mode], model_config.n1, model_config.n2, model_config.n3
    arch = f"{head}{n1}{n2}{n3}"
    
    # Build the hyperbolic parameters string if needed
    hyp_params = ""
    if 'h' in arch:
        hyp_params = f"_k{model_config.curvature}"
        if model_config.attn_k_lr:
            hyp_params += f"_attn_lr{model_config.attn_k_lr:.0e}"  # Using shorter scientific notation
        if model_config.head_k_lr:
            hyp_params += f"_head_lr{model_config.head_k_lr:.0e}"  # Using shorter scientific notation
    
    # Combine all components
    run_id = f"{seconds_since_midnight}_{dataset_aliases[dataset_name]}_{arch}{hyp_params}_s{train_config.seed}"
    return date, run_id


def setup_tokenizer(train_config: TrainConfig, model_config: MultiGeomGPTConfig): # Tokenizer setup
    if "tinystories_char" in train_config.train_bin_path:
        dataset_name = "TinyStoriesChar"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tinystories_char/char_tokenizer.json")
    elif "tinystories" in train_config.train_bin_path:
        dataset_name = "TinyStories"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(train_config.train_bin_path, "tinystories_tokenizer.json"),
            eos_token="<|endoftext|>",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
    elif "wikitext2" in train_config.train_bin_path:
        dataset_name = "Wikitext2"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(train_config.train_bin_path[:-9], "wikitext2_tokenizer.json"),
            eos_token="<|endoftext|>",
            unk_token="<UNK>",
            pad_token="<PAD>",
        )
    elif "wikitext103" in train_config.train_bin_path:
        dataset_name = "Wikitext103"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(train_config.train_bin_path[:-9], "wikitext103_tokenizer.json"),
            eos_token="<|endoftext|>",
            unk_token="<UNK>",
            pad_token="<PAD>",
        )
    else:
        dataset_name = "FineWeb"
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
    
    model_config.vocab_size = tokenizer.vocab_size

    return dataset_name, tokenizer, model_config


def encode_text(tokenizer: Union[GPT2TokenizerFast, PreTrainedTokenizerFast], text: str, device: str):
    """Encodes a string into token IDs."""
    return tokenizer.encode(text, return_tensors="pt").to(device)


def decode_tokens(tokenizer: Union[GPT2TokenizerFast, PreTrainedTokenizerFast], tokens: torch.Tensor, train_config: TrainConfig):
    """Decodes token IDs into a readable string."""
    # For character-level tokenizer, join characters without spaces
    if "tinystories_char" in train_config.train_bin_path:
        return ''.join(tokenizer.convert_ids_to_tokens(tokens.cpu().tolist()))
    # For word-level tokenizers, use normal decoding
    return tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)


def setup_ddp():
    """Setting up distributed env"""
    assert torch.cuda.is_available(), "CUDA is required for DDP but not available."

    # import os, torch
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("torch.cuda.device_count():", torch.cuda.device_count())


    try:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable for DDP: {e}")
    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Map local rank to CUDA device
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)

    print(f"[Rank {ddp_rank}] Using device: {device}")

    # Identify master process
    master_process = (ddp_rank == 0)
    if master_process:
        print(f"[Rank {ddp_rank}] This is the master process.")
    

    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def compute_grad_norm(params):
    """Compute the total L2 norm of gradients in the given list of parameters."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)  # L2 norm
            # if torch.isnan(param_norm):
            #     print()
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5


# Train and model config initialization
train_config = TrainConfig()
model_config = MultiGeomGPTConfig()

# Set seeds
random.seed(train_config.seed)
np.random.seed(train_config.seed)
torch.manual_seed(train_config.seed)
torch.cuda.manual_seed_all(train_config.seed)

# Tokenizer setup
dataset_name, tokenizer, model_config = setup_tokenizer(train_config, model_config)

# DDP setup
ddp_is_enabled = "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
if ddp_is_enabled:
    ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()
else:
    ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = 0, 0, 1, train_config.device, True
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.cuda.set_device(device)

# Convenience variables
B, T = train_config.device_batch_size, train_config.context_length
assert train_config.num_val_tokens % (B * T * ddp_world_size) == 0, "num_val_tokens must be divisible by num tokens processed by all devices simultaniously."
val_steps = train_config.num_val_tokens // (B * T * ddp_world_size)

assert train_config.global_batch_size % (B * ddp_world_size) == 0, "global_batch_size must be divisible by device_batch_size * ddp_world_size."
train_accumulation_steps = train_config.global_batch_size // (B * ddp_world_size)


# Dataloaders
train_loader = DistributedDataLoader(train_config.train_bin_path, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(train_config.val_bin_path, B, T, ddp_rank, ddp_world_size)

# Just logging in master process
if master_process:
    print(f"Training DataLoader: {train_loader.ntok_total / 1e6:.2f}M tokens across {len(train_loader.files)} files.")
    print(f"Validation DataLoader: {val_loader.ntok_total / 1e6:.2f}M tokens across {len(val_loader.files)} files.")
    print(f"train_config.num_val_tokens / val_loader.ntok_total = {train_config.num_val_tokens / val_loader.ntok_total:.2f}")
x, y = train_loader.next_batch()

# Model setup and wrap to DDP
model = MultiGeometryGPT(model_config).to(device)
# x = torch.randint(low=0, high=model_config.vocab_size - 1, size=(train_config.device_batch_size, model_config.context_length)).to(device)
# result = model(x)
# model = model.to(device)    
# model = torch.compile(model)
if ddp_is_enabled:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model: MultiGeometryGPT = model.module  # Always access raw model via .module
else:
    raw_model = model

# Optional: Verify that DDP is correctly set up
if master_process:
    print(f"[Rank {ddp_rank}] Model wrapped in DDP.")

# bfloat16 autocast context
amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# Gather curvature parameters
head_k_params, attn_k_params = [], []
for name, param in raw_model.named_parameters():
    if "lm_head.k" in name:
        if model_config.head_k_lr > 0:
            param.requires_grad = True
            head_k_params.append(param)
        else:
            param.requires_grad = False
    elif "curvature" in name:
        if model_config.attn_k_lr > 0:
            param.requires_grad = True
            attn_k_params.append(param)
        else:
            param.requires_grad = False

k_params = head_k_params + attn_k_params

# Again logging in master process
if master_process:
    print(f"learnable k params lengths: head = {len(head_k_params)}, attn = {len(attn_k_params)}")
    print(f"Tokenizer vocab size: {model_config.vocab_size}")

# Configure optimizers
lm_head_params = [p for name, p in raw_model.lm_head.named_parameters() if (p.requires_grad and ("k" not in name))]

params = [p for name, p in raw_model.transformer.layers.named_parameters() if (p.requires_grad and ("curvature" not in name))]
matrix_params = [p for p in params if p.ndim == 2]
vector_params = [p for p in params if p.ndim == 1]
wte_params = [raw_model.transformer.wte.weight]

optimizer_head = torch.optim.Adam(lm_head_params, lr=model_config.head_lr, betas=(0.9, 0.999), eps=1e-10, fused=True)
optimizer_wte = torch.optim.Adam(wte_params, lr=model_config.wte_lr, betas=(0.9, 0.999), eps=1e-10, fused=True)
# optimizer_matrix = Muon(matrix_params, lr=model_config.matrix_lr, momentum=0.95, ddp_is_enabled=ddp_is_enabled)
optimizer_matrix = torch.optim.Adam(matrix_params, lr=model_config.matrix_lr, betas=(0.9, 0.999), eps=1e-10, fused=True)

opt = Prodigy(lm_head_params + wte_params + matrix_params, lr=1., use_bias_correction=True, weight_decay=0.0)
optimizers = [optimizer_head, optimizer_matrix, optimizer_wte]
total_params_list = lm_head_params + wte_params + matrix_params
if len(vector_params) > 0:
    optimizer_vector = torch.optim.Adam(vector_params, lr=model_config.vector_lr, betas=(0.9, 0.999), eps=1e-10, fused=True)
    optimizers.append(optimizer_vector)

    total_params_list += vector_params

# optimizers = [Prodigy(total_params_list, lr=1., use_bias_correction=True, weight_decay=0.0)]

if len(attn_k_params) > 0 and len(head_k_params) > 0:
    # optimizer_k = torch.optim.SGD([
    #     {"params": head_k_params, "lr": model_config.head_k_lr},  
    #     {"params": attn_k_params, "lr": model_config.attn_k_lr}  
    # # ], fused=True)
    # ], momentum=0.9, nesterov=True)
    total_params_list += attn_k_params
    total_params_list += head_k_params
    # optimizers.append(optimizer_k)
    if master_process:
        print(f"block curvatures are learned with lr={model_config.attn_k_lr}")
        print(f"lm_head.k is learned with lr={model_config.head_k_lr}")

elif len(attn_k_params) > 0 and len(head_k_params) == 0:
    # optimizer_k = torch.optim.SGD([
    #     {"params": attn_k_params, "lr": model_config.attn_k_lr}
    # # ], fused=True)
    # ], momentum=0.9, nesterov=True)
    total_params_list += attn_k_params
    # optimizers.append(optimizer_k)
    if master_process:
        print(f"block_curvatures are learned with {model_config.attn_k_lr} lr")

elif len(attn_k_params) == 0 and len(head_k_params) > 0:
    # optimizer_k = torch.optim.SGD([
    #     {"params": head_k_params, "lr": model_config.head_k_lr}
    # # ], fused=True)
    # ], momentum=0.9, nesterov=True)
    total_params_list += head_k_params
    # optimizers.append(optimizer_k)
    if master_process:
        print(f"lm_head.k is learned with {model_config.head_k_lr} lr")

else:
    if master_process:
        print(f"block curvatures and lm_head.k are not learned")

optimizers = [Prodigy(total_params_list, lr=1., use_bias_correction=True, weight_decay=0.0)]

# Configure schedulers
def get_lr(it):
    t = max(0, min(1, 1 - it / train_config.num_iterations))
    w = min(t / train_config.cooldown_frac, 1.0)
    return w * train_config.init_lr + (1 - w) * train_config.end_lr
    
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Again logging in master process
if master_process:
    model_size = raw_model.model_size()
    print("\n=== Model ===")
    print(f"Model Size:    {model_size}\n")
    print(f"Train Bin Path:            {train_config.train_bin_path}")
    print(f"Val Bin Path:            {train_config.val_bin_path}")
    print(f"Context Length:      {train_config.context_length}")
    print(f"Batch Size (global):  {train_config.global_batch_size}")
    print(f"Batch Size (device):  {train_config.device_batch_size}")
    print(f"n_layers:              {model_config.n_layers}")
    print(f"n_heads:               {model_config.n_heads}")
    print(f"head_dim:             {model_config.head_dim}")
    print(f"n_embd:               {model_config.n_embd}")
    print("\n=== Experiment ===")
    print(f"Head mode:             {model_config.lm_head_mode}")
    print(f"Init curvature:        {model_config.curvature}")
    print(f"Head curvature learning rate: {model_config.head_k_lr}")
    print(f"Attn curvature learning rate: {model_config.attn_k_lr}")
    print(f"Seed:                 {train_config.seed}")
    print("==============================\n")

# Print model parameters
if master_process:
    print("=" * 30 + "MODEL PARAMETERS" + "=" * 30)
    for name, param in raw_model.named_parameters():
        print(f"param:{name}, requires_grad={param.requires_grad}")
    print("="*76)

# And again logging in master process
if master_process:
    # Create the run ID
    now = datetime.datetime.now()
    date, run_id = create_run_id(train_config, model_config, dataset_name, now)
    # Create log directory and file
    logdir = f'runs/{date}/{run_id}/'
    checkpoints_dir = 'ckpts/'
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "tensorboard_logs"), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Logs for this run will be stored in: {logdir}")

    print("Writing logs to: " + os.path.join(logdir, "tensorboard_logs"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tensorboard_logs"))

    writer.add_text("train_config", pretty_json(vars(train_config)))
    writer.add_text("model_config", pretty_json(vars(model_config)))
    writer.add_text("model_size", model_size)
    logfile = os.path.join(logdir, 'log.txt')
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('=' * 100 + '\n')
        f.write(code)
        f.write('=' * 100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')


# Train loop begin here
training_time_s = 0.0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
total_t0 = time.time()
train_loss_accum = 0.0
train_loss_count = 0
# begin training
train_loader.reset()
if not model_config.multi_geom_block:
    raw_model.normalize_sph_matrices()

for step in tqdm(range(train_config.num_iterations + 1)):
    last_step = (step == train_config.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_s = 0.0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (train_config.val_loss_every > 0 and step % train_config.val_loss_every == 0)):
        print("=" * 20 + "VALIDATION STARTED" + "=" * 20)
        # stop the clock
        torch.cuda.synchronize()
        training_time_s += time.time() - t0
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in tqdm(range(val_steps), position=1, leave=False):
            x_val, y_val = val_loader.next_batch()
            with amp_ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                logits, loss = model(x_val, y_val)
                val_loss += loss.detach()
                del loss
        if ddp_is_enabled:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            tokens_seen = step * train_config.global_batch_size * train_config.context_length
            print(f'step:{step}/{train_config.num_iterations}, tokens seen: {tokens_seen / 1e6:.2f}M, val_loss:{val_loss:.4f} train_time:{training_time_s:.2f}s step_avg:{1000 * training_time_s / (timed_steps - 1):.0f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{train_config.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_s:.2f}s step_avg:{1000 * training_time_s / (timed_steps - 1):.0f}ms\n')
            writer.add_scalar('Loss/Validation', val_loss.item(), tokens_seen)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if train_config.generate_every and master_process and ((step) % train_config.generate_every == 0):
        # Use a fixed prompt or context for generation
        prompt = "London is a "  # Customize as per your dataset
        context = encode_text(tokenizer, prompt, device)
        
        # Generate text
        generated_tokens = raw_model.generate(context, max_length=50, temperature=1.0, top_k=50)
        generated_text = decode_tokens(tokenizer, generated_tokens[0], train_config)
        
        # Log the generated text to TensorBoard
        writer.add_text(f"Generated_Text/Step_{step}", generated_text, step)
        
        # Optionally log to console for immediate feedback
        print(f"[Step {step}] Generated Text: {generated_text}")

    if master_process and (last_step or (train_config.save_every > 0 and step % train_config.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_s += time.time() - t0
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, checkpoints_dir + '%s_state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with amp_ctx:
            _, loss = model(x, y)
            train_loss = loss.detach()
        train_loss_accum += train_loss.item()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps and ddp_is_enabled:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step

    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"WARNING: Parameter {name} has no gradient. Skipping.")
            continue
        p.grad /= train_accumulation_steps

    if model_config.grad_k_clip != 0.0:
        if head_k_params:
            torch.nn.utils.clip_grad_norm_(head_k_params, model_config.grad_k_clip)
        if attn_k_params:
            torch.nn.utils.clip_grad_norm_(attn_k_params, model_config.grad_k_clip)
    
    if model_config.total_grad_clip:
        torch.nn.utils.clip_grad_norm_(total_params_list, model_config.total_grad_clip)
    
    if master_process and step % train_config.train_loss_every == 0:

        grad_norm_lm_head = compute_grad_norm(lm_head_params)
        grad_norm_matrix = compute_grad_norm(matrix_params)
        grad_norm_wte = compute_grad_norm(wte_params)
        grad_norm_head_k = compute_grad_norm(head_k_params)
        grad_norm_attn_k = compute_grad_norm(attn_k_params)
        if len(vector_params) > 0:
            grad_norm_vector = compute_grad_norm(vector_params)

        writer.add_scalar("grad_norm/lm_head", grad_norm_lm_head, step)
        writer.add_scalar("grad_norm/matrix", grad_norm_matrix, step)
        writer.add_scalar("grad_norm/wte", grad_norm_wte, step)
        writer.add_scalar("grad_norm/head_k", grad_norm_head_k, step)
        writer.add_scalar("grad_norm/attn_k", grad_norm_attn_k, step)
        if len(vector_params) > 0:
            writer.add_scalar("grad_norm/vector", grad_norm_vector, step)

        print("=" * 30 + "attn k PARAMS" + "=" * 30)
        for param in attn_k_params:
            print(torch.exp(param.detach()).cpu().numpy().squeeze())
        # print("=" * 70)

        print("=" * 30 + "head k PARAMS" + "=" * 30)
        for param in head_k_params:
            print(torch.exp(param.detach()).cpu().numpy().squeeze())
        print("=" * 72)

        print(f"Step: {step}, grad_norm/lm_head", grad_norm_lm_head)
        print(f"Step: {step}, grad_norm/matrix", grad_norm_matrix)
        print(f"Step: {step}, grad_norm/wte", grad_norm_wte)
        print(f"Step: {step}, grad_norm/head_k", grad_norm_head_k)
        print(f"Step: {step}, grad_norm/attn_k", grad_norm_attn_k)
        if len(vector_params) > 0:
            print(f"Step: {step}, grad_norm/vector", grad_norm_vector)

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        # sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # train_loss_accum += train_loss.item()
    train_loss_count += train_accumulation_steps
    if not model_config.multi_geom_block:
        raw_model.normalize_sph_matrices()
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process and step % train_config.train_loss_every == 0:# within the main training loop, after logging validation loss or training loss
        
        avg_train_loss = train_loss_accum / train_loss_count
        elapsed_time = time.time() - total_t0
        approx_time = training_time_s + (time.time() - t0)
        avg_time_per_step = approx_time / timed_steps
        estimated_total_time = avg_time_per_step * train_config.num_iterations
        tokens_seen = step * train_config.global_batch_size * train_config.context_length 
        print(f"step:{step}/{train_config.num_iterations}, tokens seen:{tokens_seen / 1e6:.2f}M, avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}/{estimated_total_time:.0f}s step_avg:{1000 * avg_time_per_step:.0f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step}/{train_config.num_iterations} avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}s step_avg:{1000 * avg_time_per_step:.0f}ms\n")
        writer.add_scalar('Loss/Train', avg_train_loss, tokens_seen)
        train_loss_accum = 0.0
        train_loss_count = 0

        # Add curvature logging here
        if k_params:  # Only log if curvature is learnable
            # Log head curvature
            for i, param in enumerate(head_k_params):
                curvature_value = torch.exp(param.detach()).item()  
                if i == 0:
                    writer.add_scalar(f"Curvature/Head", curvature_value, tokens_seen)
            
            # Log attention layer curvatures
            for i, param in enumerate(attn_k_params):
                curvature_values = torch.exp(param.detach().squeeze()).cpu() # Shape: (n_heads,)
                if bool(curvature_values.size()):
                    values_str = ' '.join([f"{v:.2f}" for v in curvature_values])
                else:
                    values_str = f"{curvature_values.item()}"
                # print(f"Attn layer {i} curvatures: [{values_str}]")
                
                # Log each head's curvature to tensorboard
                if bool(curvature_values.size()):
                    for head_idx, value in enumerate(curvature_values):
                        writer.add_scalar(f"Curvature/Attn/{i}/Head_{head_idx}", value, tokens_seen)
                else:
                    writer.add_scalar(f"Curvature/Attn/{i}/Head_0", curvature_values.item(), tokens_seen)

if master_process:
    total_training_time = time.time() - total_t0
    print(f"Total training time: {total_training_time:.2f}s")
    with open(logfile, "a") as f:
        f.write(f"Total training time: {total_training_time:.2f}s\n")
    print(f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if master_process:
    writer.close()
if ddp_is_enabled:
    dist.destroy_process_group()
