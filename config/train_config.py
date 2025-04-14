from dataclasses import dataclass
from .model_config import context_length


@dataclass
class TrainConfig:
    # Dataset configuration
    train_bin_path: str = "./data/fineweb10B/fineweb_train_*.bin"
    val_bin_path: str = "./data/fineweb10B/fineweb_val_*.bin"
    context_length: int = context_length  # train context length

    # Training setup
    device: str = "cuda:0" # used only w/o DDP
    device_batch_size: int = 2
    # global_batch_size: int = 8 * 32 # global batch size must be divisible to (device_batch_size * ddp_world_size) in order to perfrom grad accumulation steps
    global_batch_size: int = 252 # for 6 devices
    num_iterations: int = 10000
    seed: int = 42
    # num_val_tokens: int = 10_485_760 # num_val_tokens must be divisible by (device_batch_size * context_length * ddp_world_size) tokens processed by all devices simultaniously
    num_val_tokens: int = 10_481_664 # for 6 devices
    # Optimization parameters
    init_lr: float = 1.0         # Base learning rate
    end_lr: float = 0.1     # Final learning rate after cooldown
    cooldown_frac: float = 0.5  # Fraction of training for LR cooldown

    # Logging and monitoring
    train_loss_every: int = 10   # Log training loss every N steps
    val_loss_every: int = 500    # Calculate validation loss every N steps
    generate_every: int = 500    # Generate samples every N steps
    gradient_log_every: int = 500 # Log gradient statistics every N steps
    save_every: int = 500 # Save checkpoint

    # Generation parameters
    generation_temp: float = 0.8    # Temperature for sampling
    top_k: int = 50               # Top-k filtering
    max_gen_length: int = 512     # Max tokens to generate

    # Precision and performance
    use_amp: bool = True           # Use automatic mixed precision
    compile_model: bool = True      # Compile model with torch.compile

    # Regularization
    dropout: float = 0.1            # Dropout probability
    weight_decay: float = 0.1       # AdamW weight decay
    grad_clip: float = 1.0          # Gradient clipping value