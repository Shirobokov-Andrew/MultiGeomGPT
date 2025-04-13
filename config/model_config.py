from dataclasses import dataclass

context_length = 512


@dataclass
class MultiGeomGPTConfig:
    multi_geom_block: bool = False
    n_layers: int = 12 # must be n_euc_layers+n_hyp_layers+n_sph_layers in case of multi_geom_block == False, and arbitrary otherwise

    n_embd: int = 768 # embedding dimension
    n_heads: int = 6 # must be n1+n2+n3 in case of multi_geom_block == True, and arbitrary otherwise
    head_dim: int = 128 # must be n_embd // n_heads

    n1: int = 2 # number of euc heads (relevant only when multi_geom_block == True)
    n2: int = 2 # number of hyp heads (relevant only when multi_geom_block == True)
    n3: int = 2 # number of sph heads (relevant only when multi_geom_block == True)

    # geom_dict: dict = {"euc": 4, "hyp": 4, "sph": 4}
    layers_order = ("euc", "sph", "hyp") # (relevant only when multi_geom_block == False)
    n_euc_layers: int = 6 # number of fully euc transformer layers (relevant only when multi_geom_block == False)
    n_hyp_layers: int = 0 # number of fully hyp transformer layers (relevant only when multi_geom_block == False)
    n_sph_layers: int = 0 # number of fully sph transformer layers (relevant only when multi_geom_block == False)

    lm_head_mode: str = 'euc'
    curvature: float = 1.0
    context_length: int = context_length

    attn_k_lr: float = 0.002
    head_k_lr: float = 0.0
    head_lr: float = 0.0022
    wte_lr: float = 0.01
    matrix_lr: float = 0.0005
    vector_lr: float = 0.001
    dropout: float = 0.0
    vocab_size: float = 65