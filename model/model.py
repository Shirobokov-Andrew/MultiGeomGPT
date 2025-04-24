import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lmath import project, distance, norm, logmap0, expmap0
from config.model_config import MultiGeomGPTConfig


class Rotary(nn.Module):
    """Rotary Positional Embeddings"""

    def __init__(self, dim: int, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()[None, :, None, :].bfloat16()
            self.sin_cached = freqs.sin()[None, :, None, :].bfloat16()
        return self.cos_cached, self.sin_cached


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
    return rotated.type_as(x)


def custom_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dropout: nn.Dropout = None,
        curvature: float = None, 
        mode: str = "euc", 
        hyp_attn_weight: str = "neg_dis",
        scale: float = None, 
        ) -> torch.Tensor:
    if mode == "euc":
        return F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale)
    elif mode == "sph":
        q_norm = F.normalize(query, p=2, dim=-1)
        k_norm = F.normalize(key, p=2, dim=-1)
        return F.scaled_dot_product_attention(q_norm, k_norm, value, is_causal=True, scale=scale)
    elif mode == "hyp":
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn_bias.masked_fill_(mask, float('-inf'))

        lq = project(query, k=curvature).unsqueeze(-2)
        lk = project(key, k=curvature).unsqueeze(-3)
        dis = distance(lq, lk, k=curvature)
        if hyp_attn_weight == "inverse_dis":
            attn_weight = query.shape[-1] ** (0.5) / (1e-6 + dis)
        elif hyp_attn_weight == "neg_dis":
            attn_weight = -dis

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout is not None:
            attn_weight = dropout(attn_weight)
        return attn_weight @ value
        

class MultiGeometryAttention(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig):
        super().__init__()
        self.config = config
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.resid_dropout = nn.Dropout(p=config.dropout)

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.rotary = Rotary(config.head_dim)
    
        if self.config.n2 > 0:
            self.init_hyperbolic_params()

    def init_hyperbolic_params(self):
        if self.config.attn_k_lr:
            self.hyp_curvature = nn.Parameter(
                torch.full((1, self.config.n2, 1, 1), self.config.curvature))
            # x = torch.randn(1, self.config.n2, 1, 1, device=self.qkv.weight.device)
            # init_k = torch.exp(x) * self.config.curvature
            # self.hyp_curvature = nn.Parameter(init_k)
        else:
            self.register_buffer('hyp_curvature',
                                    torch.full((1, self.config.n2, 1, 1), self.config.curvature))
        if self.config.learn_x0:
            self.x0_norm_proj = nn.Linear(self.config.head_dim, 1)
            if self.config.init_x0_uniform:
                nn.init.uniform_(self.x0_norm_proj.weight, a=-(self.config.head_dim ** 0.5), b=self.config.head_dim ** 0.5)
                nn.init.uniform_(self.x0_norm_proj.bias, a=-(self.config.head_dim ** 0.5), b=self.config.head_dim ** 0.5)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [x.view(B, T, self.config.n_heads, self.config.head_dim) for x in qkv]

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        split_sizes = [self.config.n1, self.config.n2, self.config.n3]
        qs = torch.split(q, split_sizes, dim=2)
        ks = torch.split(k, split_sizes, dim=2)
        vs = torch.split(v, split_sizes, dim=2)

        outputs = []
        if self.config.n1 > 0:
            attn = custom_attention(
                qs[0].transpose(1, 2), 
                ks[0].transpose(1, 2), 
                vs[0].transpose(1, 2), 
                mode='euc', 
                dropout=self.attn_dropout, 
                scale=1 / math.sqrt(self.config.head_dim),
                )
            outputs.append(attn.transpose(1, 2))
        if self.config.n2 > 0:
            if self.config.learn_x0:
                if self.config.init_x0_uniform:
                    q_norms = self.x0_norm_proj(qs[1]) ** 2
                    k_norms = self.x0_norm_proj(ks[1]) ** 2
                    q_att = q_norms * F.normalize(qs[1], dim=-1)
                    k_att = k_norms * F.normalize(ks[1], dim=-1)
                else:
                    q_norms = self.x0_norm_proj(qs[1])
                    k_norms = self.x0_norm_proj(ks[1])
                    q_att = qs[1] + q_norms * F.normalize(qs[1], dim=-1)
                    k_att = ks[1] + k_norms * F.normalize(ks[1], dim=-1)
            else:
                q_att = qs[1]
                k_att = ks[1]
            attn = custom_attention(
                q_att.transpose(1, 2), 
                k_att.transpose(1, 2), 
                vs[1].transpose(1, 2),
                curvature=self.hyp_curvature, 
                mode='hyp', 
                dropout=self.attn_dropout,
                hyp_attn_weight=self.config.hyp_attn_weight,
                )
            outputs.append(attn.transpose(1, 2))
        if self.config.n3 > 0:
            attn = custom_attention(
                qs[2].transpose(1, 2), 
                ks[2].transpose(1, 2), 
                vs[2].transpose(1, 2), 
                mode='sph', 
                dropout=self.attn_dropout, 
                scale=math.sqrt(self.config.head_dim),
                )
            outputs.append(attn.transpose(1, 2))

        x = torch.cat(outputs, dim=2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(x))


class MultiGeometryBlock(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig):
        super().__init__()
        self.attn = MultiGeometryAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.dropout)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class JointGeometryMLP(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str):
        """
        :param geom_type: geometry type - can be euc, hyp or sph
        """
        super().__init__()
        self.geom_type = geom_type
        self.config = config
        use_bias = False if geom_type == "sph" else True

        if config.use_mlp_swiglu:
            self.uv_proj = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=use_bias)
            self.act = nn.SiLU()
            self.out_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=use_bias)
        else:
            self.expand_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=use_bias)
            self.act = nn.GELU()
            self.shrink_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=use_bias)

        if geom_type == "sph" and config.use_mlp_swiglu:
            self._init_spherical_params()

    def _init_spherical_params(self):
        self.init_value = 1.0
        self.init_scaling = 1.0
        self.suv = nn.Parameter(self.init_scaling * torch.ones(2 * 4 * self.config.n_embd,))
    
    def forward(self, x: torch.Tensor):
        if self.config.use_mlp_swiglu:
            uv = self.uv_proj(x)
            if self.geom_type == "sph":
                scaling = (self.init_value / self.init_scaling) * (self.config.n_embd ** 0.5)
                uv = self.suv * scaling * uv
            u, v = torch.chunk(uv, 2, dim=-1)
            h = u * self.act(v)
            h = self.out_proj(h)
        else:
            h = self.expand_proj(x)
            h = self.act(h)
            h = self.shrink_proj(h) 

        return h


class JointGeometryAttention(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str):
        """
        :param geom_type: geometry type - can be euc, hyp or sph
        """
        super().__init__()
        self.config = config
        self.geom_type = geom_type

        self.qkv = nn.Linear(config.n_embd, 3 * config.head_dim * config.n_heads, bias=False)
        self.out_proj = nn.Linear(config.head_dim * config.n_heads, config.n_embd, bias=False)
        self.rotary = Rotary(config.head_dim)

        if geom_type == "hyp":
            self._init_hyperbolic_params()
        elif geom_type == "sph":
            self._init_spherical_params()

    def _init_hyperbolic_params(self):
        if self.config.attn_k_lr:
            x = torch.randn(1, self.config.n_heads, 1, 1, device=self.qkv.weight.device)
            init_k = torch.exp(x) * self.config.curvature
            self.hyp_curvature = nn.Parameter(init_k)
        else:
            x = torch.randn(1, self.config.n_heads, 1, 1, device=self.qkv.weight.device)
            fixed_k = torch.exp(x) * self.config.curvature
            self.register_buffer('hyp_curvature', fixed_k)
        if self.config.learn_x0:
            self.x0_norm_proj = nn.Linear(self.config.head_dim, 1)
    
    def _init_spherical_params(self):
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = 1.0 / (self.config.config.n_embd ** 0.5)
        self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.config.n_heads, self.config.head_dim) for t in qkv]

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # get learnable time component in Lorentz model
        if self.geom_type == "hyp" and self.config.learn_x0:
            q_norms = self.x0_norm_proj(q) ** 2
            k_norms = self.x0_norm_proj(k) ** 2
            q = q_norms * F.normalize(q, dim=-1)
            k = k_norms * F.normalize(k, dim=-1)

        if self.geom_type == "euc":
            attn = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mode="euc", scale=1 / (self.config.head_dim ** 0.5))
        elif self.geom_type == "hyp":
            attn = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mode="hyp", curvature=self.hyp_curvature, hyp_attn_weight=self.config.hyp_attn_weight)
        elif self.geom_type == "sph":
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, 1, self.config.n_heads, self.config.head_dim)
            q = sqk * F.normalize(q, p=2.0, dim=-1)
            k = sqk * F.normalize(k, p=2.0, dim=-1)
            attn = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mode="euc", scale=self.config.head_dim ** 0.5)

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn)


class JointGeometryBlock(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str):
        """
        :param geom_type: geometry type - can be euc, hyp or sph
        """
        super().__init__()
        assert geom_type in ["sph", "hyp", "euc"], f"Unknown geometry was given: {geom_type}"

        self.config = config
        self.geom_type = geom_type

        self.attn = JointGeometryAttention(config, geom_type)
        self.mlp = JointGeometryMLP(config, geom_type)

        if geom_type == "sph":
            self._init_spherical_params()

    def _init_spherical_params(self):
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / (self.config.n_embd ** 0.5)
        self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(self.config.n_embd))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / (self.config.n_embd ** 0.5)
        self.mlp_alpha = nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd))

    def forward(self, x: torch.Tensor):
        """
        x can be hyp/euc/sph tensor
        """
        if x.shape[-1] == self.config.n_embd:
            x = x
        elif x.shape[-1] == self.config.n_embd + 1:
            x = x.narrow(-1, 1, x.shape[-1] - 1)
        else:
            raise ValueError(f"input x has invalid vector dimension: {x.shape[-1]}")

        if self.geom_type in ["euc", "hyp"]:
            h = F.rms_norm(x, (x.size(-1),))
        else:
            h = F.normalize(x, p=2.0, dim=-1)

        h_attn = self.attn(h)

        if self.geom_type == "sph":
            x = self._sph_skip_connection(x, h_attn, "attn")
        else:
            x = x + h_attn

        if self.geom_type in ["euc", "hyp"]:
            h = F.rms_norm(x, (x.size(-1),))
        else:
            h = x

        h_mlp = self.mlp(h)

        if self.geom_type == "sph":
            x = self._sph_skip_connection(x, h_mlp, "mlp")
        else:
            x = x + h_mlp

        return x

    def _sph_skip_connection(self, x: torch.Tensor, h: torch.Tensor, place: str):
        if place == "attn":
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        else:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = F.normalize(x, p=2.0, dim=-1)
        B_norm = F.normalize(h, p=2.0, dim=-1)

        return F.normalize(A_norm + lr * (B_norm - A_norm), p=2.0, dim=-1)


class FullyHyperbolicBlock(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str = "hyp"):
        super().__init__()
        self.config = config
        self.geom_type = geom_type

        self.block_curvature = nn.Parameter(torch.tensor(config.curvature))
        self.mlp_curvature = nn.Parameter(torch.tensor(config.curvature))
        self.attn_curvatures = nn.Parameter(torch.ones(1, config.n_heads, 1, 1) * config.curvature)

        self.qkv = nn.Linear(config.n_embd + 1, 3 * config.n_embd, bias=False)
        self.attn_proj = nn.Linear((config.head_dim + 1) * config.n_heads, config.n_embd)

        self.rotary = Rotary(config.head_dim)

        self.mlp_expand = nn.Linear(config.n_embd + 1, 4 * config.n_embd + 3)
        self.act = nn.GELU()
        self.mlp_shrink = nn.Linear(4 * (config.n_embd + 1), config.n_embd)

        self.normalization = nn.LayerNorm(config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x can be hyp/euc/sph tensor
        """
        if x.shape[-1] == self.config.n_embd:
            lx = project(x, k=self.block_curvature) # map euc/sph input to block hyperboloid (B, T, n_embd + 1)
        elif x.shape[-1] == self.config.n_embd + 1:
            # x_norm = norm(x, keepdim=True) # map hyp input x to block hyperboloid (B, T, n_embd + 1)
            # lx = x * torch.sqrt(torch.exp(self.block_curvature)) / x_norm
            lx = x
        else:
            raise ValueError(f"input x has invalid vector dimension: {x.shape[-1]}")

        attn_lx = self.attn(self.block_norm(lx)) # (B, T, n_embd+1)
        lx = self.hyp_skip_connection(lx, attn_lx) # (B, T, n_embd+1)
        mlp_lx = self.mlp(self.block_norm(lx)) # (B, T, n_embd+1)
        lx = self.hyp_skip_connection(lx, mlp_lx) # (B, T, n_embd+1)

        return lx

    def attn(self, lx: torch.Tensor) -> torch.Tensor:
        """
        lx is a tensor from hyperboloid (B, T, n_embd + 1)
        """
        B, T, C = lx.shape[0], lx.shape[1], lx.shape[2]
        # calculate q, k, v from hyperbolic vector and map them to another hyperboloid
        qkv = self.qkv(lx).split(C - 1, dim=2) # here C==n_embd+1, since lx is hyperbolic
        q, k, v = [x.view(B, T, self.config.n_heads, self.config.head_dim) for x in qkv] # (B, T, num_heads, head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        lq = project(q.transpose(1, 2), k=self.attn_curvatures).unsqueeze(-2) # (B, num_heads, T, 1, head_dim+1) # unsqueeze to calculate pairwise distance
        lk = project(k.transpose(1, 2), k=self.attn_curvatures).unsqueeze(-3) # (B, num_heads, 1, T, head_dim+1)
        lv = project(v.transpose(1, 2), k=self.attn_curvatures) # (B, num_heads, T, head_dim+1)

        attn_bias = torch.zeros(T, T, dtype=lq.dtype, device=lq.device)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=lq.device), diagonal=1)
        attn_bias.masked_fill_(mask, float('-inf'))

        # attn matrix is an determined by the inverse geodesic distance on hyperboloid
        dis = distance(lq, lk, k=self.attn_curvatures) # (B, num_heads, T, T)
        if self.config.hyp_attn_weight == "inverse_dis":
            attn_weight = lq.shape[-1] ** (0.5) / (1e-6 + dis)
        elif self.config.hyp_attn_weight == "neg_dis":
            attn_weight = -dis
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1) # (B, num_heads, T, T)

        # we need to renormalize attention in order to get vector on the initial attn hyperboloid
        attn = attn_weight @ lv  # (B, num_heads, T, head_dim+1)
        attn_hyp_norm = norm(attn, keepdim=True)
        attn = attn * torch.sqrt(torch.exp(self.attn_curvatures)) / attn_hyp_norm
        attn = attn.transpose(1, 2).contiguous().view(B, T, (self.config.head_dim + 1) * self.config.n_heads)

        # project from attention hyperboloids to euc space and back to block hyperboloid
        attn = self.attn_proj(attn)
        attn = project(attn, k=self.block_curvature) # (B, T, n_embd+1)

        return attn

    def hyp_skip_connection(self, lx: torch.Tensor,  f_lx: torch.Tensor) -> torch.Tensor:
        """
        lx and f_lx are tensors from hyperboloid (B, T, n_embd + 1)
        """
        skipped = project(lx.narrow(-1, 1, lx.shape[-1] - 1) + f_lx.narrow(-1, 1, f_lx.shape[-1] - 1), k=self.block_curvature)
        # skipped = expmap0(logmap0(lx, k=self.block_curvature) + logmap0(f_lx, k=self.block_curvature), k=self.block_curvature)
        # skipped_norm = norm(skipped, keepdim=True)
        # skipped = skipped * torch.sqrt(torch.exp(self.block_curvature)) / skipped_norm

        return skipped
    
    def mlp(self, lx: torch.Tensor) -> torch.Tensor:
        """
        lx is a tensor from hyperboloid (B, T, n_embd + 1)
        """
        lx = self.mlp_expand(lx) # (B, T, 4*n_embd + 3)
        lx = project(lx, k=self.mlp_curvature) # (B, T, 4*n_embd + 4)
        lx = lx.narrow(-1, 1, lx.shape[-1] - 1) # (B, T, 4*n_embd + 3)
        lx = self.act(lx) # (B, T, 4*n_embd + 3)
        lx = project(lx, k=self.mlp_curvature) # (B, T, 4*n_embd + 4)
        lx = self.mlp_shrink(lx) # (B, T, n_embd)
        lx = project(lx, k=self.block_curvature) # (B, T, n_embd + 1)

        return lx
    
    def block_norm(self, lx: torch.Tensor) -> torch.Tensor:
        """
        lx is a tensor from hyperboloid (B, T, n_embd + 1)
        """
        lx = lx.narrow(-1, 1, lx.shape[-1] - 1)
        # lx = F.rms_norm(lx, (lx.size(-1),))
        lx = self.normalization(lx)
        lx = project(lx, k=self.block_curvature)

        return lx


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model"""

    def __init__(self, head_k_lr: float, init_k: float, vocab_size: int, n_embd: int):
        super().__init__()
        if head_k_lr > 0:
            self.k = nn.Parameter(torch.tensor(init_k))
        else:
            self.register_buffer('k', torch.tensor(init_k))
        self.a = torch.nn.Parameter(torch.zeros(vocab_size, ))  # optimize properly
        self.z = torch.nn.Parameter(F.pad(torch.zeros(vocab_size, n_embd - 2), pad=(1, 0), value=1)) # (vocab_size, n_embd-1)

        self.init_weights()

    def forward(self, x: torch.Tensor):
        # x: (B, T, num_features)

        # Hyperplane parameters
        # x = project(x, k=self.k)
        sqrt_mK = 1 / self.k.sqrt()  # scalar
        norm_z = torch.norm(self.z, dim=-1)  # (vocab_size,)
        w_t = torch.sinh(sqrt_mK * self.a) * norm_z  # (vocab_size,)
        w_s = torch.cosh(sqrt_mK * self.a).unsqueeze(-1) * self.z  # (vocab_size, num_features-1)

        beta = torch.sqrt(-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2)  # (vocab_size,)

        x0 = x.narrow(-1, 0, 1)  # (B, T, 1)
        x_rest = x.narrow(-1, 1, x.shape[-1] - 1)  # (B, T, num_features-1)
        inner_prod = torch.matmul(x_rest, self.z.T)  # (B, T, vocab_size)
        alpha = -x0 * w_t.view(1, 1, -1) + torch.cosh(sqrt_mK * self.a).view(1, 1, -1) * inner_prod  # (B, T, vocab_size)
        sqrt_mK_alpha_over_beta = sqrt_mK * alpha / beta.view(1, 1, -1)
        d = self.k.sqrt() * torch.abs(torch.asinh(sqrt_mK_alpha_over_beta))  # (B, T, vocab_size)

        logits = torch.sign(alpha) * beta.view(1, 1, -1) * d  # (B, T, vocab_size)

        return logits

    def init_weights(self):
        stdv = 1. / math.sqrt(1 + self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class MultiGeometryGPT(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig):
        super().__init__()
        self.config = config

        wte = nn.Embedding(config.vocab_size, config.n_embd)

        if config.multi_geom_block:
            layers = nn.ModuleList([MultiGeometryBlock(config) for _ in range(config.n_layers)])
        else:
            geom_dict = {"euc": config.n_euc_layers, "hyp": config.n_hyp_layers, "sph": config.n_sph_layers}
            layers_list = []
            self.layers_geoms = []

            for geom_type in config.layers_order:
                n_layers = geom_dict[geom_type]
                for _ in range(n_layers):
                    self.layers_geoms.append(geom_type)
                if geom_type == "hyp" and config.use_fully_hyp_block:
                    layers_list.extend([FullyHyperbolicBlock(config) for _ in range(n_layers)])
                else:
                    layers_list.extend([JointGeometryBlock(config, geom_type) for _ in range(n_layers)])
            
            layers = nn.ModuleList(layers_list)

        self.transformer = nn.ModuleDict(dict(wte=wte, layers=layers))

        if isinstance(layers_list[-1], FullyHyperbolicBlock):
            n_embd = config.n_embd + 1
        else:
            n_embd = config.n_embd

        if config.lm_head_mode == 'euc':
            self.lm_head = nn.Linear(n_embd, config.vocab_size)
            self._init_weights(self.lm_head)
        elif config.lm_head_mode == 'hyp':
            self.lm_head = LorentzMLR(config.head_k_lr, config.curvature, config.vocab_size, n_embd)
        elif config.lm_head_mode == 'sph':
            self.lm_head = nn.Linear(n_embd, config.vocab_size)
            self._init_weights(self.lm_head)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.transformer.wte(idx)

        for block in self.transformer.layers:
            x = block(x)
        
        # if the last layer is fully hyperbolic we have to reduce dimension by 1 to get n_embd dimension
        # if x.shape[-1] == self.config.n_embd + 1:
        #     x = x.narrow(-1, 1, x.shape[-1] - 1)
        
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    def normalize_sph_matrices(self):
        for layer in self.transformer.layers:
            if layer.geom_type == "sph":
                layer.attn.qkv.weight.data.copy_(F.normalize(layer.attn.qkv.weight.data, p=2.0, dim=-1))
                layer.attn.out_proj.weight.data.copy_(F.normalize(layer.attn.out_proj.weight.data, p=2.0, dim=-1))

                if self.config.use_mlp_swiglu:
                    layer.mlp.uv_proj.weight.data.copy_(F.normalize(layer.mlp.uv_proj.weight.data, p=2.0, dim=-1))
                    layer.mlp.out_proj.weight.data.copy_(F.normalize(layer.mlp.out_proj.weight.data, p=2.0, dim=-1))
                else:
                    layer.mlp.expand_proj.weight.data.copy_(F.normalize(layer.mlp.expand_proj.weight.data, p=2.0, dim=-1))
                    layer.mlp.shrink_proj.weight.data.copy_(F.normalize(layer.mlp.shrink_proj.weight.data, p=2.0, dim=-1))

    def generate(self, context: torch.Tensor, max_length: int, temperature: float = 1.0, top_k: int = None):
        self.eval()
        generated = context.clone()
        for _ in range(max_length):
            with torch.no_grad():
                logits, _ = self(generated[:, -self.config.context_length:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
    
    def model_size(self):
        """Calculate the model size in millions or thousands, based on parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        else:
            return f"{total_params / 1e3:.2f}K"


if __name__ == "__main__":
    config = MultiGeomGPTConfig(multi_geom_block=False)
    model = MultiGeometryGPT(config=config)
    x = torch.randint(low=0, high=config.vocab_size - 1, size=(16, config.context_length))
    result = model(x)